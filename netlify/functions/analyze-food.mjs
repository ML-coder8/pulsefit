// Estimates the nutrition content of a food photo using Claude's vision model.
//
// The browser posts a base64 data URL; we return a fixed JSON shape that maps
// directly onto the fields the scanner UI already renders. The API key stays
// here - index.html is served publicly and must never hold it.

import Anthropic, { APIConnectionError, APIError, RateLimitError } from '@anthropic-ai/sdk';
import { zodOutputFormat } from '@anthropic-ai/sdk/helpers/zod';
import { z } from 'zod';

// Field names deliberately match the client's FOOD_DATABASE entries
// (cals/protein/carbs/fat) so no translation layer is needed.
const FoodAnalysis = z.object({
  isFood: z.boolean().describe('False if the photo contains no identifiable food or drink.'),
  name: z.string().describe('Short dish name, e.g. "Chicken Caesar Salad".'),
  serving: z.string().describe('Estimated portion visible, e.g. "1 bowl (approx. 320g)".'),
  cals: z.number().describe('Total calories for the portion shown, in kcal.'),
  protein: z.number().describe('Grams of protein for the portion shown.'),
  carbs: z.number().describe('Grams of carbohydrate for the portion shown.'),
  fat: z.number().describe('Grams of fat for the portion shown.'),
  confidence: z.number().describe('Confidence in this estimate, 0-100.'),
});

const SYSTEM_PROMPT = `You are a nutrition estimation engine for a calorie tracking app.

Given a photo of food, identify the dish and estimate its nutrition for the
portion actually visible in the image - not a generic serving size. Use visual
cues such as plate diameter, cutlery, and hands for scale.

Guidelines:
- Estimate the portion in the photo. A half-eaten plate is not a full serving.
- Macros should be roughly consistent with the calorie total
  (protein 4 kcal/g, carbs 4 kcal/g, fat 9 kcal/g). Being within ~15% is fine.
- Set confidence honestly. A clear photo of a single dish warrants 85-95.
  An ambiguous, dark, or partially hidden dish warrants 50-70.
- If the image contains no food or drink, set isFood to false, confidence to 0,
  all numbers to 0, and name to a brief description of what you actually see.`;

const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
const ALLOWED_MEDIA_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
  });
}

export default async (req) => {
  if (req.method !== 'POST') {
    return json({ error: 'method_not_allowed' }, 405);
  }

  if (!process.env.ANTHROPIC_API_KEY) {
    // Surfaced distinctly so the client can tell misconfiguration apart from
    // a genuine analysis failure.
    return json({ error: 'missing_api_key' }, 503);
  }

  let imageDataUrl;
  try {
    ({ imageDataUrl } = await req.json());
  } catch {
    return json({ error: 'invalid_json' }, 400);
  }

  if (typeof imageDataUrl !== 'string') {
    return json({ error: 'missing_image' }, 400);
  }

  const match = imageDataUrl.match(/^data:([^;]+);base64,(.+)$/);
  if (!match) {
    return json({ error: 'malformed_data_url' }, 400);
  }

  const [, mediaType, base64Data] = match;

  if (!ALLOWED_MEDIA_TYPES.includes(mediaType)) {
    return json({ error: 'unsupported_media_type', mediaType }, 415);
  }

  // base64 encodes 3 bytes per 4 characters.
  if (Math.floor(base64Data.length * 0.75) > MAX_IMAGE_BYTES) {
    return json({ error: 'image_too_large' }, 413);
  }

  const client = new Anthropic();

  try {
    const response = await client.messages.parse({
      model: 'claude-opus-5',
      max_tokens: 16000,
      system: SYSTEM_PROMPT,
      thinking: { type: 'adaptive' },
      output_config: {
        effort: 'medium',
        format: zodOutputFormat(FoodAnalysis),
      },
      messages: [
        {
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: mediaType, data: base64Data } },
            { type: 'text', text: 'Identify this food and estimate its nutrition for the portion shown.' },
          ],
        },
      ],
    });

    if (response.stop_reason === 'refusal') {
      return json({ error: 'refused', category: response.stop_details?.category ?? null }, 422);
    }

    const analysis = response.parsed_output;
    if (!analysis) {
      return json({ error: 'unparsable_response' }, 502);
    }

    if (!analysis.isFood) {
      return json({ error: 'not_food', saw: analysis.name }, 422);
    }

    return json({
      name: analysis.name,
      serving: analysis.serving,
      cals: Math.round(analysis.cals),
      protein: Math.round(analysis.protein),
      carbs: Math.round(analysis.carbs),
      fat: Math.round(analysis.fat),
      confidence: Math.round(analysis.confidence),
    });
  } catch (err) {
    console.error('Food analysis failed:', err);

    if (err instanceof RateLimitError) {
      return json({ error: 'rate_limited' }, 429);
    }
    if (err instanceof APIConnectionError) {
      return json({ error: 'upstream_unreachable' }, 504);
    }
    if (err instanceof APIError) {
      return json({ error: 'upstream_error', status: err.status }, 502);
    }
    return json({ error: 'analysis_failed' }, 500);
  }
};
