import { lrEn }      from './lr.js';
import { logitEn }   from './logit.js';
import { dtEn }      from './dt.js';
import { nbEn }      from './nb.js';
import { svmEn }     from './svm.js';
import { knnEn }     from './knn.js';
import { clusterEn } from './cluster.js';
import { nnEn }      from './nn.js';
import { cnnEn }     from './cnn.js';
import { gnnEn }     from './gnn.js';

export const enBlogs = [...lrEn, ...logitEn, ...dtEn, ...nbEn, ...svmEn, ...knnEn, ...clusterEn, ...nnEn, ...cnnEn, ...gnnEn];

// Fast slug → English blog lookup
export const enBlogMap = Object.fromEntries(enBlogs.map((b) => [b.slug, b]));
