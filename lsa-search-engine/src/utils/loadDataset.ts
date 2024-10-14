export async function loadLSAData() {
    const response = await fetch('/lsa_metadata.json');
    if (!response.ok) {
      throw new Error('Failed to load metadata');
    }
    return await response.json();
  }