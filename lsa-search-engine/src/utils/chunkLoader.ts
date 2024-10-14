export async function loadChunk(chunkIndex: number) {
  const response = await fetch(`/chunks/chunk_${chunkIndex}.json`);
  if (!response.ok) {
    throw new Error(`Failed to load chunk ${chunkIndex}`);
  }
  const chunkData = await response.json();
  return chunkData;
}