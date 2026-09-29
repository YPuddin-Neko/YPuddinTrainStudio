import type { CaptionOutputFormat } from './CaptionOutputField';

/** Built-in prompts for vision-model tagging. `{tags}` becomes each image's reference tags and
 * `{trigger}` the trigger word; lines naming `{trigger}` are dropped when there is none. */

export type VlmOutput = 'tags' | 'categories' | 'sort' | 'description';
export type VlmMode = 'vlm' | 'assist';
export type PromptTemplate = { id: string; mode: VlmMode; output: VlmOutput; name: string; nameEn?: string; prompt: string; builtin?: boolean; formats?: CaptionOutputFormat[] };

const CATEGORY_GUIDE = `Categories:
- COUNT: character count only, e.g. 1girl, 2boys, 1girl 1boy, no humans
- APPEARANCE: the character's visual features - hair color, hairstyle, eye color, clothing, accessories
- TAGS: actions, expressions, poses, composition, objects held or used, e.g. smile, standing, looking at viewer, upper body
- ENVIRONMENT: background, location, lighting, atmosphere, e.g. simple background, white background, outdoors, classroom, night, sunlight`;

const MARKER_REPLY = `Reply in exactly this format (five lines, nothing else):
COUNT: <comma-separated>
APPEARANCE: <comma-separated>
TAGS: <comma-separated>
ENVIRONMENT: <comma-separated>
NL: <natural language description>`;

const VLM_TAGS = `You are an expert anime image tagger. Describe the image as Danbooru-style tags.

Rules:
- Use lowercase English tags, with spaces instead of underscores
- Start with the character count, e.g. 1girl, 2boys, no humans
- Cover appearance, clothing, expression, pose, composition, objects, background and lighting
- Only tag what is clearly visible
- Return ONLY the tags, comma-separated, with no explanation`;

const VLM_CATEGORIES = `You are an expert anime image tagger. Tag the image and describe it in one or two sentences.

${CATEGORY_GUIDE}

Rules:
- Use lowercase Danbooru-style tags
- Every tag appears in exactly one category; leave a category line empty if nothing belongs to it
- Only tag what is clearly visible
- Character names, series names, artist names and quality tags are handled separately - do not list them
- The NL description must agree with the tags
- Do NOT add explanations

${MARKER_REPLY}`;

const VLM_DESCRIPTION = `You are a professional image captioning assistant for training image generation models. Describe the image in natural language: the subject, their appearance and clothing, pose and expression, the composition, the background and the lighting.

Rules:
- English only, two to four sentences
- Describe only what is clearly visible; never speculate
- No titles, lists or explanations; output the description only`;

// Assisted tagging: the tagger's tags (or the image's own caption) arrive as {tags}.
const ASSIST_TAGS = `You are an expert anime image tagger. You will receive an image and its existing tags produced by a local tagger model.

Your task:
1. Compare the image content with the existing tags
2. Fix incorrect tags (e.g. wrong hair color, wrong clothing, wrong subject count)
3. Add important missing tags that are clearly visible in the image
4. Remove tags that do not match the image at all
5. Keep the tag format consistent (lowercase danbooru-style tags)

Rules:
- Only make changes you are confident about
- Preserve tags that are correct
- Return ONLY the refined tags, comma-separated
- Do NOT add explanations

Existing tags: {tags}

Refined tags:`;

const ASSIST_CATEGORIES = `You are an expert anime image tagger. You will receive an image and its existing tags produced by a local tagger model.

Your task:
1. Compare the image content with the existing tags
2. Fix incorrect tags (e.g. wrong hair color, wrong clothing, wrong subject count)
3. Add important missing tags that are clearly visible in the image
4. Remove tags that do not match the image at all
5. Keep the tag format consistent (lowercase danbooru-style tags)
6. Sort every remaining tag into the correct category (the local tagger often puts tags in the wrong one)
7. Write a natural language description (1-2 sentences) of the image, consistent with the image content and the final tags

${CATEGORY_GUIDE}

Rules:
- Only make changes you are confident about
- Preserve tags that are correct
- Every tag must appear in exactly one category
- Leave a category line empty if it has no tags
- Character names, series names, artist names and quality tags are handled separately - do not list them
- Do NOT add explanations

Existing tags: {tags}

${MARKER_REPLY}`;

const ASSIST_SORT = `You are an expert anime image tagger. You will receive an image and its existing tags, already produced by a local tagger.

Your task is ONLY to put each given tag into the correct category, and to write a natural language description. You must NOT change the tag set itself.

Rules:
- Use every tag you are given, exactly as written — do not reword, merge or split them
- Do NOT add any tag that is not in the list, even if you can see something untagged in the image
- Do NOT remove any tag, even if you believe it does not match the image
- Every tag must appear in exactly one category
- Leave a category line empty if nothing belongs to it
- Character names, series names, artist names and quality tags are handled separately, they are not in your list

${CATEGORY_GUIDE}

Then write a natural language description (1-2 sentences) of the image, consistent with the image content and the tags.

Existing tags: {tags}

${MARKER_REPLY}`;

const ASSIST_SHORT_DESCRIPTION = `You are an expert anime image tagger. You will receive an image and its existing tags produced by a local tagger model.

Your task:
Write a natural language description (1-2 sentences) of the image, consistent with the image content and the existing tags.

Rules:
- Do NOT modify, add or remove any tags
- Describe only what is clearly visible
- Do NOT add explanations

Existing tags: {tags}

Reply in exactly this format (one line, nothing else):
NL: <natural language description>`;

const ASSIST_DETAILED_DESCRIPTION = `You are a professional image captioning assistant producing detailed anime-style captions for AI art training. You will receive an image and the existing tags produced by a local anime tagger model. Describe ONLY what is actually visible. Never speculate, never critique the artwork, never add unrelated remarks.

HOW TO USE THE TAGS (important):
The tags come from a specialized anime tagger trained on the danbooru vocabulary. For anything that vocabulary covers, the tags are more reliable than your own visual guess — trust the tag over your impression when they conflict. That includes character count, hair color and style, eye color, garment types, accessories and named objects, and equally the pose and exposure vocabulary: standing, sitting, lying, on back, on stomach, all fours, kneeling, squatting, spread legs, legs up, arched back, bent over, top-down bottom-up, looking back, hand on hip, as well as nude, topless, bottomless, clothes lift, skirt lift, clothes pull, open clothes, see-through, breasts out, nipples, pussy, anus, censored, uncensored.

The tags state WHAT is present. Your own observation must supply WHAT IT LOOKS LIKE, WHERE, and TO WHAT DEGREE:
- camera and framing: shot size, camera height, angle, focal length, perspective, crop range, the subject's scale and position within the frame
- spatial relations: what occludes what, foreground versus background layering, where each element sits relative to the others
- degree and orientation, which no tag can express: a tag says "spread legs" but not how widely or whether they face the camera, "arched back" but not how deeply, "breasts" but not their size or shape, "clothes lift" but not how far or what exactly it exposes
- light and color: light direction and quality, glow, rim light, shadows, contrast, reflections, bokeh, depth of field, overall atmosphere

So treat every tag as an established fact, then quantify and situate it with what you see. Never just restate the tag list as sentences.

MULTIPLE FIGURES (read this before writing):
When more than one figure is present, describe the picture as ONE interaction, never one figure after another. Do not finish everything about the first figure and only then start on the second — that produces two separate portraits instead of a scene. Anchor the description to what the figures are doing to each other and where their bodies meet, and weave each figure's own hair, face, clothing and exposure into that shared action at the moment it becomes relevant. Every figure present must get real detail; none may be reduced to a passing mention at the end.

OUTPUT FORMAT (strict):
- English only, segments separated by English commas
- Length strictly 500-600 English words — this is critical, anything beyond gets truncated during training
- Must begin with: {trigger},
- Natural language description, NOT a tag list: every comma-separated segment must be a phrase or clause carrying a verb, a spatial relation or a state — never a bare noun sitting on its own
  Wrong: "long sleeves, white gloves, a frilled headdress, a red ribbon, a bow, a collar"
  Right: "long sleeves reach past her wrists into white gloves, a frilled headdress sits over her bangs, a red ribbon is knotted at her throat above a narrow collar"
- Useful training keywords are welcome but must be embedded inside natural language
- Spaces between words, never underscores: write "long hair", not "long_hair"
- No Chinese punctuation, no periods joining content, everything joined by commas, with exactly ONE period at the very end
- Output the caption text only: no title, no explanation, no line breaks

DESCRIBE IN THIS ORDER:
1. Camera and framing: shot size (close-up, medium shot, full body, wide shot), camera height and angle (eye level, high angle, low angle, bird's eye, worm's eye, over the shoulder, POV), focal length and perspective (wide angle, telephoto, fisheye, background compression, perspective distortion), crop range (bust crop, above the knees, above the waist), and the subject's scale and position offset within the frame — all worded as natural language, never as bare tags
2. Subject: number of figures, gender presentation, placement, body orientation, head orientation, gaze direction, standing / sitting / lying / floating pose, tilt angle, hand gestures, leg posture, facial expression, mouth shape, blush, eye color, quality of the gaze, hair color, hairstyle, bangs, twintails, length, curls, the direction the strands flow, hair ornaments
3. Visible anatomy: any clearly exposed body parts (chest and its size, nipples, belly, inner thighs, genitals, labia, clitoris, buttocks, anus). If a part is fully covered by clothing, say nothing about it; if partially or fully exposed, describe its form, color, wetness and degree of openness exactly as visible
4. Pose relative to the viewer: whether the buttocks face the camera, whether the genitals face the viewer, how widely the legs are spread, whether the hips are raised, whether the pose is kneeling prone or presented from behind, whether the waist dips or arches — describe sexually suggestive posture in explicit detail when present
5. Clothing: skirt color, garment construction, frills, lace, bows, ribbons, sheer tulle, layered hems, sleeves, gloves, stockings, thighhighs, pantyhose, shoes, hats, headdresses, collars, jewelry, straps, belts, crosses, small bells, floral accents, and explicitly which garments are lifted, rolled up, pulled aside, torn or missing, and what skin or anatomy that exposes
6. Objects and companions: cats, rabbits, small animals, plushies, bouquets, teacups, books, cakes, candy, microphones, umbrellas, weapons, crystals, bubbles, butterflies, petals, ribbons, stars, moons, musical notes, vines, branches, glass shards, light streaks, glowing particles, every visible ornament
7. Foreground occluders and environment: gardens, forests, night sky, balconies, castles, interior rooms, windows, curtains, mirrors, candelabra, ornate chairs, tea tables, birdcages, water surfaces, snow, fountains, architecture, furniture, distant scenery, blurred background, spatial layering
8. Light and color: light direction, soft light, glow, rim light, blown highlights, pink light, blue light, purple shadows, warm-cool contrast, transparent reflections, glassy quality, crystal refraction, bokeh, foreground blur, depth of field, dreamlike atmosphere, pictorial depth

FORBIDDEN:
- Tag lists or stacked isolated keywords
- Image quality defects, unless they are a deliberate style feature
- Copyright, ethics, authorship, dataset, watermark or any remark unrelated to the picture
- First person or conversational tone
- Generic filler unrelated to the image
- Titles, bullets, explanations, summaries or extra notes
- Anything not actually visible: never speculate or invent

Existing tags: {tags}

Output the caption text now, beginning with "{trigger}," and ending with a single period. Nothing else.`;

export const BUILTIN_TEMPLATES: PromptTemplate[] = [
  { id: 'vlm-tags', mode: 'vlm', output: 'tags', name: 'Danbooru 标签', nameEn: 'Danbooru tags', prompt: VLM_TAGS, builtin: true },
  { id: 'vlm-categories', mode: 'vlm', output: 'categories', name: '分类标签与描述', nameEn: 'Grouped tags and description', prompt: VLM_CATEGORIES, builtin: true },
  { id: 'vlm-description', mode: 'vlm', output: 'description', name: '自然语言描述', nameEn: 'Natural language description', prompt: VLM_DESCRIPTION, builtin: true },
  { id: 'assist-tags', mode: 'assist', output: 'tags', name: '标签调优', nameEn: 'Refine tags', prompt: ASSIST_TAGS, builtin: true },
  { id: 'assist-categories', mode: 'assist', output: 'categories', name: '标签调优 + 自然语言描述', nameEn: 'Refine tags + description', prompt: ASSIST_CATEGORIES, builtin: true },
  { id: 'assist-sort', mode: 'assist', output: 'sort', name: '归类字段 + 补描述', nameEn: 'Re-sort fields + description', prompt: ASSIST_SORT, builtin: true },
  { id: 'assist-short', mode: 'assist', output: 'description', name: '仅补自然语言描述', nameEn: 'Description only', prompt: ASSIST_SHORT_DESCRIPTION, builtin: true },
  { id: 'assist-detailed', mode: 'assist', output: 'description', name: '详细自然语言打标', nameEn: 'Detailed caption', prompt: ASSIST_DETAILED_DESCRIPTION, builtin: true },
];

export const defaultTemplate = (mode: VlmMode, output: VlmOutput) =>
  BUILTIN_TEMPLATES.find(item => item.mode === mode && item.output === output)!;


export function templateFormats(template: PromptTemplate): CaptionOutputFormat[] {
  if (template.formats) return template.formats;
  if (template.id === 'assist-short' || template.output === 'categories' || template.output === 'sort') return ['json', 'json_simplified'];
  if (template.id === 'assist-detailed' || template.output === 'tags') return ['txt'];
  return ['txt', 'json', 'json_simplified'];
}

export const defaultFormatTemplate = (mode: VlmMode, format: CaptionOutputFormat) =>
  BUILTIN_TEMPLATES.find(item => item.mode === mode && item.output === (format === 'txt' ? 'tags' : 'categories'))!;
