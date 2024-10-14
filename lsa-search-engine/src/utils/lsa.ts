import * as tf from '@tensorflow/tfjs';
import { preprocessText } from './textProcessing';

export function processQuery(query: string, terms: string[], Vt: number[][]) {
  const queryTokens = preprocessText(query);
  const queryVector = new Array(terms.length).fill(0);
  queryTokens.forEach((token) => {
    const index = terms.indexOf(token);
    if (index !== -1) {
      queryVector[index] += 1;
    }
  });

  if (queryVector.every((value) => value === 0)) {
    throw new Error('No matching terms in query.');
  }

  const queryTensor = tf.tensor1d(queryVector);
  const Vt_tf = tf.tensor2d(Vt);

  // Correctly project the query into latent space
  const queryLatent = tf.dot(queryTensor, Vt_tf.transpose());

  return queryLatent;
}
