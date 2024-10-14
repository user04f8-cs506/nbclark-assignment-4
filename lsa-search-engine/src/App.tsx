import React, { useState, useEffect } from 'react';
import { loadLSAData } from './utils/loadDataset';
import { loadChunk } from './utils/chunkLoader';
import { processQuery } from './utils/lsa';
import { Bar } from 'react-chartjs-2';
import * as tf from '@tensorflow/tfjs';
import { FaSearch } from 'react-icons/fa';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface LSAMetadata {
  // S: number[];
  Vt: number[][];
  terms: string[];
  U: number[][];
  documents: { text: string }[];
}

interface SearchResult {
  index: number;
  similarity: number;
  document: { text: string };
}

const NUM_CHUNKS = 189;

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [chartData, setChartData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [chunksLoaded, setChunksLoaded] = useState(0);
  const [totalChunks, setTotalChunks] = useState(NUM_CHUNKS); // Total number of chunks
  const [lsaData, setLsaData] = useState<LSAMetadata>({
    // S: [],
    Vt: [],
    terms: [],
    U: [],
    documents: [],
  });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const initialize = async () => {
      try {
        const metadata = await loadLSAData();
        setLsaData({ ...metadata, U: [], documents: [] });
        await loadNextChunk(0); // Start loading the first chunk
      } catch (err: unknown) {
        if (err instanceof Error) {
          setError(err.message);
        } else {
          setError('An unknown error occurred while loading metadata.');
        }
      }
    };

    initialize();
  }, []);

  const loadNextChunk = async (chunkIndex: number) => {
    try {
      console.log(`Loading chunk ${chunkIndex}...`);
      const chunkData = await loadChunk(chunkIndex);
      console.log(`Chunk ${chunkIndex} loaded.`);

      setLsaData((prevData) => ({
        ...prevData,
        U: prevData.U.concat(chunkData.U),
        documents: prevData.documents.concat(chunkData.documents),
      }));
      setChunksLoaded(chunkIndex + 1);

      if (chunkIndex + 1 < totalChunks) {
        await loadNextChunk(chunkIndex + 1);
      } else {
        setLoading(false);
        console.log("All chunks loaded.");
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unknown error occurred during chunk loading.');
      }
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      setError('Please enter a query.');
      return;
    }

    setError(null); // Clear any previous error messages

    try {
      const { U, Vt, terms, documents } = lsaData;
      const queryLatent = processQuery(query, terms, Vt);

      const U_tf = tf.tensor2d(U);
      // Compute the dot product between document vectors and query vector
      const similarities = tf.matMul(U_tf, queryLatent.expandDims(1)).squeeze();

      // Compute the norms of document vectors (rows of U_tf)
      const docNorms = tf.norm(U_tf, 'euclidean', 1); // Euclidean norm along rows

      // Compute the norm of the query vector
      const queryNorm = tf.norm(queryLatent);

      // Compute the product of norms and add a small epsilon to avoid division by zero
      const normsProduct = docNorms.mul(queryNorm).add(tf.scalar(1e-8));

      // Compute cosine similarities
      const cosineSimilarities = similarities.div(normsProduct);

      const similaritiesArray = (await cosineSimilarities.array()) as number[];
      const topResults: SearchResult[] = similaritiesArray
        .map((similarity, index) => ({
          index,
          similarity,
          document: documents[index],
        }))
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 5);

      setResults(topResults);
      setChartData({
        labels: topResults.map((res) => `Doc ${res.index}`),
        datasets: [
          {
            label: 'Cosine Similarity',
            data: topResults.map((res) => res.similarity),
            backgroundColor: 'rgba(75, 192, 192, 0.6)',
          },
        ],
      });

      // Clean up tensors
      U_tf.dispose();
      similarities.dispose();
      cosineSimilarities.dispose();
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unknown error occurred during query processing.');
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 via-pink-500 to-red-500 p-8 flex items-center justify-center">
      <div className="relative max-w-2xl w-full">
        <div className="absolute inset-0 bg-gradient-to-r from-white to-transparent opacity-20 rounded-2xl filter blur-lg"></div>
        <div className="relative bg-white bg-opacity-10 border border-white border-opacity-30 backdrop-filter backdrop-blur-xl rounded-2xl shadow-2xl p-8">
          {loading ? (
            <div className="flex flex-col items-center justify-center min-h-screen">
              <div className="text-white text-xl mb-4">Loading data...</div>
              <div className="w-64 bg-gray-300 rounded-full h-4">
                <div
                  className="bg-blue-500 h-4 rounded-full"
                  style={{ width: `${(chunksLoaded / totalChunks) * 100}%` }}
                ></div>
              </div>
              <div className="text-white mt-2">
                {Math.round((chunksLoaded / totalChunks) * 100)}%
              </div>
            </div>
          ) : (
            <>
              <div className="text-center mb-12">
                <h1 className="text-5xl font-bold text-white mb-4">LSA Search Engine</h1>
                <p className="text-xl text-gray-200">
                  Discover relevant documents using Latent Semantic Analysis.
                </p>
              </div>
              
              <input
                type="text"
                className="w-full p-4 rounded mb-6 bg-white bg-opacity-30 placeholder-gray-200 text-white focus:outline-none focus:ring-2 focus:ring-purple-300 transition duration-300"
                placeholder="Type your query here..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              
              <button
                onClick={handleSearch}
                className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-pink-500 hover:to-purple-500 transition duration-300 transform hover:scale-105 flex items-center justify-center"
              >
                <FaSearch className="mr-2" />
                Search
              </button>
              
              {error && <div className="text-red-500 text-center mt-4">{error}</div>}
              
              {results.length > 0 && (
                <div className="mt-8">
                  <h2 className="text-3xl font-semibold text-white mb-6">Results</h2>
                  <ul>
                    {results.map((result, idx) => (
                      <li key={idx} className="mb-6">
                        <div className="bg-white bg-opacity-20 p-4 rounded-lg shadow-lg transition duration-300 hover:bg-opacity-30">
                          <p className="text-white text-lg font-semibold">
                            Document {result.index}: Similarity {result.similarity.toFixed(4)}
                          </p>
                          <p className="text-gray-200 mt-2">
                            {result.document.text.substring(0, 200)}...
                          </p>
                        </div>
                      </li>
                    ))}
                  </ul>
                  <div className="mt-8">
                    <Bar
                      key={JSON.stringify(chartData)} // This ensures a new chart is created when data changes
                      data={chartData}
                      options={{
                        responsive: true,
                        scales: {
                          x: {
                            ticks: { color: 'white' },
                            grid: { display: false }
                          },
                          y: {
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                          }
                        },
                        animation: {
                          duration: 1000,
                          easing: 'easeInOutQuad'
                        }
                      }}
                    />
                  </div>
                </div>
              )}
              
              <footer className="mt-16 text-center text-gray-300">
                &copy; 2024 Nathan Clark. All rights reserved.
              </footer>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;