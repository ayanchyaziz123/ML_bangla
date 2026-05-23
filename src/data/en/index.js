import { lrEn }    from './lr.js';
import { logitEn } from './logit.js';
import { dtEn }    from './dt.js';
import { nbEn }    from './nb.js';
import { svmEn }   from './svm.js';

export const enBlogs = [...lrEn, ...logitEn, ...dtEn, ...nbEn, ...svmEn];

// Fast slug → English blog lookup
export const enBlogMap = Object.fromEntries(enBlogs.map((b) => [b.slug, b]));
