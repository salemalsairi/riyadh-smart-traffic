# SCATS-Style Traffic Control Simulation: Riyadh (Zone A)

A data engineering project that reallocates traffic-signal green time across three
Riyadh arterials, then measures how much delay that reallocation could save. The whole
pipeline runs on data I collected myself, not a tutorial dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Status](https://img.shields.io/badge/status-complete-2EC27E)

---

## The problem

Almost everyone in Riyadh drives, and the road network has not kept up. The result is
heavy congestion at peak hours, longer trips, and more fuel burned while cars sit idling.
A KAPSARC study on the city puts the fuel penalty from congestion as high as 29%.

Fixed-timing signals make this worse. They give the same green time to a road whether it
is jammed or empty. SCATS (Sydney Coordinated Adaptive Traffic System) is the real-world
answer: it shifts green time toward whichever approach is busier. This project simulates
that idea on a small, real slice of Riyadh and asks a concrete question. If the signals
had reacted to demand, how much delay would have gone away?

## What this does

The project is built in four stages, each one its own step in the repo.

**1. Data collection.** A Python script pulls travel time and delay every 15 minutes from
the TomTom API for 8 arterial links, grouped into three corridors in south-east Riyadh:
Batha, Firyan, and Prince Mohammed bin Abdulrahman. That gave 651 snapshots over 8 days,
including a weekend.

**2. The algorithm.** For each snapshot, the engine works out how "loaded" each corridor
is, then splits the signal cycle in proportion to that load, with a safety floor so no
approach ever gets starved. It compares the resulting delay against a naive equal split.

**3. Local RAG.** A retrieval system links the numbers back to published research. It
chunks the KAPSARC paper, embeds it, stores the vectors in a local database, and lets
Llama 3 answer questions grounded in that source. Everything runs offline through Ollama.

**4. Dashboard.** A Streamlit app turns the results into something you can click through:
a signal-timing simulator, before/after delay curves, and a heatmap that shows the sampling
gaps honestly instead of hiding them.

## How the algorithm works

The data is descriptive. It tells me travel time and delay, but not how many cars passed.
So I could not compute a true degree of saturation, and I did not pretend to. Instead I
used a proxy:

```
congestion = delay / (travel_time - delay)
```

The denominator is the free-flow travel time, so a road with no delay scores zero. This
proxy is a direct transform of the Travel Time Index used in a peer-reviewed Saudi study
on school-zone traffic, which is what gives it some footing beyond my own judgment.

Green time is then handed out in proportion to each corridor's demand share, clipped to a
12% minimum per approach. To score it, I model delay on a corridor as `demand² / green_share`,
a convex form where delay climbs faster than linearly once demand outruns the green time it
gets. Under that model, the demand-proportional split is provably the one that minimizes
total delay, which happens to be the exact principle SCATS runs on.

## Results

Across all 651 snapshots, the smart split cut the total delay index by about 34.7% against
an equal split.

The more interesting finding is where that saving comes from. I expected the gains to peak
during rush hour. They did not. The benefit tracks *imbalance* between corridors, not the
time of day. When one corridor spikes while the others sit quiet, reallocation helps a lot.
When everything is jammed together at rush hour, the corridors are more balanced and there
is less to move. The correlation between imbalance and improvement came out at 0.994, which
is about as clean as this kind of relationship gets.

Put plainly: the algorithm earns its keep when the roads are lopsided, not when they are
uniformly busy.

## The RAG layer

One detail worth calling out, because it cost me time and taught me the most. My first
embedding model was `nomic-embed-text`, and retrieval on Arabic text came back nonsense.
Rather than guess, I isolated it. I checked the database was populated, checked the model
matched on both ends, then ran the same query in English and in Arabic. English worked,
Arabic did not. That pinned the fault on the embedding model's weak Arabic, so I swapped in
`bge-m3`, a multilingual model, and rebuilt the index. Retrieval started working.

The generation step also covers for imperfect retrieval. Even when the exact "29%" passage
is not the top hit, Llama 3 reads all the returned chunks and finds it. That is the point
of RAG, and watching it happen made the idea click.

## Tech stack

- **Python**, pandas, numpy for the data and the algorithm
- **TomTom API** for the raw traffic data
- **LangChain**, ChromaDB, and Ollama (Llama 3 + bge-m3) for the RAG system
- **Streamlit** and Plotly for the dashboard

## Project structure

```
SCATS/
├── data_collector.py     # stage 1: pulls traffic data from TomTom
├── scats_engine.py       # stage 2: the allocation algorithm
├── chunking.py           # stage 3a: splits source docs into chunks
├── indexing.py           # stage 3b: builds the vector database (run once)
├── retrieval.py          # stage 3c: retrieves + generates an answer
├── dashboard.py          # stage 4: the Streamlit app
├── scats_results.csv     # algorithm output, read by the dashboard
├── study_area.jpg        # map of Zone A
└── kapsarc_db/           # persisted Chroma vectors
```

## Running it

The algorithm and dashboard need only pandas, numpy, streamlit, and plotly:

```bash
pip install pandas numpy streamlit plotly
```

Generate the results, then launch the dashboard:

```bash
python scats_engine.py --input riyadh_traffic_links.csv --output scats_results.csv
streamlit run dashboard.py
```

The RAG side needs [Ollama](https://ollama.com) running locally, with two models pulled:

```bash
ollama pull llama3
ollama pull bge-m3
python indexing.py      # build the vector store once
python retrieval.py     # ask a question
```

## Data and honesty notes

I would rather undersell this than overstate it, so a few caveats belong up front.

- **The congestion figure is a proxy, not a measured saturation.** The data has no vehicle
  counts, so I substituted `delay / free-flow time` and said so wherever it matters.
- **The 34.7% is relative and illustrative.** It comes out of a convex delay model and a
  chosen safety floor, not a field trial. Real SCATS deployments usually report something
  closer to 10 to 20%. The number is meant to show the mechanism, not promise a result.
- **Collection was periodic, not continuous.** The device sampled every 15 minutes with a
  handful of outages, mostly overnight or midday. The dashboard heatmap leaves those gaps
  blank instead of papering over them.

## Sources

- KAPSARC, *Traffic Congestion in Cities and Its Impact on Fuel Consumption Using IoT Data:
  A Riyadh Case Study* (KS-2024-DP72, 2025). Same TomTom data source, links congestion to a
  29% fuel increase.
- Shokry, Alrashidi, and Elbany, *Analyzing the Traffic Operational Performance of School
  Pick-Up and Drop-Off Dynamics in Saudi Arabia*, Sustainability 16 (2024). Source of the
  Travel Time Index my congestion proxy is derived from.

## About

I built this during my foundation year in data science and AI, mostly to prove to myself
that I could take a problem from raw data collection all the way to a working product,
without a tutorial holding my hand. I wrote the algorithm and the RAG system by hand so I
would actually understand them. The dashboard is the part I let myself polish.

If you have feedback, I want to hear it, including the critical kind.
