install:
	cd lsa-search-engine && npm install && python fetch_dataset.py

run:
	cd lsa-search-engine && npm run dev
