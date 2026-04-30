import { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { Loader2 } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

function renderInlineMarkdown(text) {
  if (!text) {
    return '';
  }

  const escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  return escaped
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*(?!\*)(.+?)\*(?!\*)/g, '$1<em>$2</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>');
}

function ResultsView({ query, useAgent, defaultQuery, agentAvailable, onSearch, onSelectMovie }) {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(() => Boolean((query || '').trim()));
  const [error, setError] = useState(null);
  const requestIdRef = useRef(0);

  useEffect(() => {
    const normalized = (query || '').trim();
    if (!normalized) {
      return;
    }

    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;

    axios
      .get('/api/search', {
        params: { text: normalized, use_agent: agentAvailable ? useAgent : false },
      })
      .then((response) => {
        if (requestIdRef.current === requestId) {
          setResults(response.data);
        }
      })
      .catch((err) => {
        if (requestIdRef.current === requestId) {
          setError('Failed to fetch search results. Please ensure the backend is running.');
          console.error(err);
        }
      })
      .finally(() => {
        if (requestIdRef.current === requestId) {
          setLoading(false);
        }
      });
  }, [query, useAgent, agentAvailable]);

  const refinedQuery = results?.query_used && results.query_used !== query ? results.query_used : null;
  const effectiveUseAgent = agentAvailable ? useAgent : false;

  const searchAgain = useMemo(
    () => () => onSearch({ text: '', useAgent: effectiveUseAgent }),
    [onSearch, effectiveUseAgent],
  );

  return (
    <div className="results-page">
      {loading && (
        <div className="page-card loading-card">
          <div className="loading-row">
            <Loader2 className="spinner loading-spinner" size={24} />
            <div>
              <div className="mini-label results-gap-xs">Working</div>
              <div className="body-copy">
                Please wait while CineSeek retrieves candidates, reranks them, and prepares the final results.
              </div>
            </div>
          </div>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      <AnimatePresence mode="wait">
        {!loading && results && (
          <motion.section
            key={`${query}-${useAgent ? '1' : '0'}`}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            className="results-shell"
          >
            <div className="results-layout">
              <aside className="results-side">
                <div className="page-card dark-panel">
                  <div className="eyebrow dark-kicker">Search Summary</div>
                  <h2 className="section-title dark-title">Results for your query</h2>
                  <div className="mono dark-query">{query || defaultQuery}</div>

                  {refinedQuery && (
                    <div className="summary-block">
                      <div className="mini-label dark-label">Refined query</div>
                      <div className="body-copy dark-copy">{refinedQuery}</div>
                    </div>
                  )}

                  {results.agent_summary && (
                    <div className="summary-block">
                      <div className="mini-label dark-label">LLM Agent</div>
                      <div className="mini-label dark-label subtle">Backend: {results.agent_model}</div>
                      <div
                        className="body-copy dark-copy"
                        dangerouslySetInnerHTML={{ __html: renderInlineMarkdown(results.agent_summary) }}
                      />
                    </div>
                  )}

                  {results.agent_error && (
                    <div className="summary-block">
                      <div className="mini-label dark-label">Fallback</div>
                      <div className="body-copy dark-copy">{results.agent_error}</div>
                    </div>
                  )}

                  <button type="button" className="back-link" onClick={searchAgain}>
                    Run another search
                  </button>
                </div>
              </aside>

              <div className="results-main">
                <div className="eyebrow results-gap-sm">Top Results</div>
                <div className="result-list">
                  {results.recommendations.map((movie, index) => (
                    <MovieCard
                      key={`${movie.title}-${index}`}
                      movie={movie}
                      rank={index + 1}
                      onSelectMovie={() =>
                        onSelectMovie({
                          title: movie.title,
                          useAgent: effectiveUseAgent,
                        })
                      }
                    />
                  ))}
                </div>
              </div>
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      <style
        dangerouslySetInnerHTML={{
          __html: `
            .results-page {
              display: flex;
              flex-direction: column;
              gap: 22px;
            }

            .results-gap-xs {
              margin-bottom: 8px;
            }

            .results-gap-sm {
              margin-bottom: 12px;
            }

            .loading-card {
              padding: 22px 24px;
            }

            .loading-row {
              display: flex;
              align-items: center;
              gap: 14px;
            }

            .spinner {
              animation: spin 1s linear infinite;
            }

            .loading-spinner {
              color: var(--accent);
            }

            .results-shell {
              display: block;
            }

            .results-layout {
              display: grid;
              grid-template-columns: minmax(280px, 0.68fr) minmax(0, 1.32fr);
              gap: 20px;
              align-items: start;
            }

            .results-side {
              position: sticky;
              top: 20px;
            }

            .dark-panel {
              background: rgba(24, 34, 47, 0.94);
              color: #f8f4ee;
            }

            .dark-kicker,
            .dark-title,
            .dark-label {
              color: #f8f4ee;
            }

            .dark-label.subtle {
              opacity: 0.72;
              margin-top: 4px;
            }

            .dark-query {
              margin-top: 10px;
              color: rgba(255, 255, 255, 0.88);
              line-height: 1.7;
              white-space: normal;
            }

            .summary-block {
              margin-top: 24px;
              padding-top: 20px;
              border-top: 1px solid rgba(255, 255, 255, 0.12);
            }

            .dark-copy {
              color: rgba(255, 255, 255, 0.82);
            }

            .dark-copy code {
              padding: 0 5px;
              border-radius: 6px;
              background: rgba(255, 255, 255, 0.08);
              font-size: 0.92em;
            }

            .back-link {
              margin-top: 28px;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-weight: 700;
              color: #fff;
              text-align: left;
              padding: 0;
            }

            .back-link:hover {
              opacity: 0.82;
            }

            .results-main {
              min-width: 0;
            }

            .result-list {
              display: grid;
              gap: 18px;
            }

            .error-message {
              color: #8f1d1d;
              background: #fbe9e9;
              border: 1px solid rgba(143, 29, 29, 0.12);
              padding: 16px 18px;
              border-radius: 18px;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
            }

            @keyframes spin {
              from { transform: rotate(0deg); }
              to { transform: rotate(360deg); }
            }

            @media (max-width: 980px) {
              .results-layout {
                grid-template-columns: 1fr;
              }

              .results-side {
                position: static;
              }
            }
          `,
        }}
      />
    </div>
  );
}

function MovieCard({ movie, rank, onSelectMovie }) {
  const info = movie.structured || {};

  return (
    <article className="page-card movie-card">
      <div className="movie-head">
        <span className="result-rank">{rank}</span>
        <div className="movie-heading">
          <button type="button" className="result-title-link" onClick={onSelectMovie}>
            {movie.title}
          </button>
        </div>
      </div>

      <div className="movie-grid">
        <div className="poster-column">
          {movie.poster_url ? (
            <img src={movie.poster_url} alt={`Poster for ${movie.title}`} className="result-poster" loading="lazy" />
          ) : (
            <div className="result-poster result-poster-placeholder">No poster</div>
          )}
        </div>

        <div className="meta-column">
          {info.overview && <p className="body-copy movie-overview">{info.overview}</p>}

          {info.genres?.length > 0 && (
            <div className="chip-block">
              <div className="mini-label results-gap-xs">Genres</div>
              <div className="chip-row">
                {info.genres.map((genre) => (
                  <span key={genre} className="info-chip neutral">
                    {genre}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="movie-columns">
            {info.director && (
              <div>
                <div className="mini-label results-gap-xs">Director</div>
                <div className="body-copy compact-copy">{info.director}</div>
              </div>
            )}

            {info.actors?.length > 0 && (
              <div>
                <div className="mini-label results-gap-xs">Actors</div>
                <div className="body-copy compact-copy">{info.actors.join(', ')}</div>
              </div>
            )}
          </div>

          {info.tags?.length > 0 && (
            <div className="chip-block">
              <div className="mini-label results-gap-xs">Tags</div>
              <div className="chip-row">
                {info.tags.map((tag) => (
                  <span key={tag} className="info-chip accent">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <style
        dangerouslySetInnerHTML={{
          __html: `
            .movie-card {
              padding: 24px 26px;
            }

            .movie-head {
              display: flex;
              align-items: center;
              gap: 14px;
              margin-bottom: 18px;
            }

            .result-rank {
              width: 42px;
              height: 42px;
              border-radius: 999px;
              background: rgba(198, 93, 46, 0.1);
              color: var(--accent-dark);
              display: inline-flex;
              align-items: center;
              justify-content: center;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-weight: 800;
              font-size: 1.05rem;
            }

            .movie-heading {
              min-width: 0;
            }

            .result-title-link {
              padding: 0;
              color: var(--ink);
              font-size: clamp(1.65rem, 2.3vw, 2.35rem);
              line-height: 1.02;
              text-align: left;
            }

            .result-title-link:hover {
              color: var(--accent-dark);
            }

            .movie-grid {
              display: grid;
              grid-template-columns: 160px minmax(0, 1fr);
              gap: 24px;
            }

            .poster-column {
              display: flex;
              align-items: flex-start;
            }

            .result-poster {
              width: 100%;
              max-width: 160px;
              aspect-ratio: 2 / 3;
              object-fit: cover;
              border-radius: 18px;
              box-shadow: 0 18px 32px rgba(24, 34, 47, 0.12);
            }

            .result-poster-placeholder {
              background: rgba(24, 34, 47, 0.08);
              color: var(--muted);
              display: flex;
              align-items: center;
              justify-content: center;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
            }

            .movie-overview {
              margin-top: 0;
              margin-bottom: 18px;
            }

            .movie-columns {
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 18px 24px;
              margin: 20px 0;
            }

            .compact-copy {
              margin: 0;
            }

            .chip-block + .chip-block {
              margin-top: 18px;
            }

            .chip-row {
              display: flex;
              flex-wrap: wrap;
              gap: 10px;
            }

            .info-chip {
              border-radius: 999px;
              padding: 8px 12px;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-size: 0.9rem;
              line-height: 1.1;
              border: 1px solid transparent;
            }

            .info-chip.neutral {
              background: rgba(255, 255, 255, 0.72);
              border-color: rgba(24, 34, 47, 0.12);
              color: var(--ink);
            }

            .info-chip.accent {
              background: rgba(198, 93, 46, 0.1);
              border-color: rgba(198, 93, 46, 0.15);
              color: var(--accent-dark);
            }

            @media (max-width: 780px) {
              .movie-grid,
              .movie-columns {
                grid-template-columns: 1fr;
              }

              .result-title-link {
                font-size: 1.8rem;
              }

              .poster-column {
                justify-content: flex-start;
              }
            }
          `,
        }}
      />
    </article>
  );
}

export default ResultsView;
