import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { Loader2, ArrowLeft } from 'lucide-react';
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
    .replace(/\*\*(.+?)\*\//g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*(?!\*)(.+?)\*(?!\*)/g, '$1<em>$2</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>');
}

const listContainerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.05,
    },
  },
};

const cardItemVariants = {
  hidden: { opacity: 0, y: 24 },
  show: {
    opacity: 1,
    y: 0,
    transition: {
      type: 'spring',
      stiffness: 260,
      damping: 24,
    },
  },
};

function ResultsView({ query, useAgent, defaultQuery, agentAvailable, onSelectMovie }) {
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

  const searchAgain = () => {
    window.location.assign('/search');
  };

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
                <div className="eyebrow results-gap-sm">Semantic Interpretation</div>
                <div className="page-card dark-panel">
                  <div className="panel-content">
                    <h2 className="section-title dark-title">Results for your query</h2>
                    <div className="mono dark-query">“{query || defaultQuery}”</div>

                    {refinedQuery && (
                      <div className="summary-block">
                        <div className="mini-label dark-label">Refined query</div>
                        <div className="body-copy dark-copy">{refinedQuery}</div>
                      </div>
                    )}

                    {results.agent_summary && (
                      <div className="summary-block ai-summary-block">
                        <div className="mini-label dark-label">LLM Agent</div>
                        <div className="agent-badge">
                          <span className="agent-badge-dot"></span>
                          <span className="agent-badge-text">{results.agent_model}</span>
                        </div>
                        <div
                          className="body-copy dark-copy ai-summary-copy"
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

                    <button type="button" className="back-btn" onClick={searchAgain}>
                      <ArrowLeft size={16} />
                      <span>Run another search</span>
                    </button>
                  </div>
                </div>
              </aside>

              <div className="results-main">
                <div className="eyebrow results-gap-sm">Top Results</div>
                <motion.div
                  className="result-list"
                  variants={listContainerVariants}
                  initial="hidden"
                  animate="show"
                >
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
                </motion.div>
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
              padding: 24px 28px;
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
              grid-template-columns: minmax(280px, 0.72fr) minmax(0, 1.28fr);
              gap: 24px;
              align-items: start;
            }

            .results-side {
              position: sticky;
              top: 24px;
            }

            .dark-panel {
              background: rgba(250, 248, 245, 0.72);
              backdrop-filter: blur(16px);
              -webkit-backdrop-filter: blur(16px);
              color: var(--ink);
              border: 1px solid rgba(24, 34, 47, 0.06);
              box-shadow: 0 20px 48px rgba(24, 34, 47, 0.04);
              position: relative;
              overflow: hidden;
              padding: 24px 26px;
            }

            .dark-panel::before {
              content: '';
              position: absolute;
              top: -30px;
              left: -30px;
              right: -30px;
              bottom: -30px;
              background: 
                radial-gradient(circle at 30px 30px, rgba(255, 235, 222, 0.85) 0%, rgba(255, 235, 222, 0.2) 50%, transparent 80%),
                radial-gradient(circle at 100% 100%, rgba(24, 34, 47, 0.04) 0%, transparent 60%);
              filter: blur(36px);
              -webkit-filter: blur(36px);
              z-index: 1;
              pointer-events: none;
              transform: translate3d(0, 0, 0); /* Force GPU acceleration */
            }

            .dark-panel::after {
              content: '';
              position: absolute;
              top: 0;
              left: 0;
              right: 0;
              bottom: 0;
              background: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.02'/%3E%3C/svg%3E");
              z-index: 2;
              pointer-events: none;
            }

            .panel-content {
              position: relative;
              z-index: 3;
            }


            .dark-title {
              color: var(--ink);
              font-size: 1.5rem;
              margin: 10px 0 0 0;
              letter-spacing: -0.01em;
              font-weight: 800;
            }

            .dark-query {
              margin-top: 10px;
              color: var(--ink);
              opacity: 0.9;
              line-height: 1.6;
              font-size: 1.1rem;
              font-style: italic;
              font-family: Georgia, serif;
            }

            .summary-block {
              margin-top: 18px;
              padding-top: 16px;
              border-top: 1px solid rgba(24, 34, 47, 0.06);
            }

            .ai-summary-block {
              border-radius: 20px;
              background: rgba(198, 93, 46, 0.03);
              border: 1px solid rgba(198, 93, 46, 0.08);
              padding: 16px;
              margin-top: 18px;
              position: relative;
              overflow: hidden;
            }

            .agent-badge {
              display: inline-flex;
              align-items: center;
              gap: 8px;
              background: rgba(198, 93, 46, 0.08);
              border: 1px solid rgba(198, 93, 46, 0.15);
              border-radius: 99px;
              padding: 4px 10px;
              margin-top: 8px;
              margin-bottom: 12px;
            }

            .agent-badge-dot {
              width: 6px;
              height: 6px;
              border-radius: 99px;
              background: var(--accent);
              box-shadow: 0 0 6px var(--accent);
            }

            .agent-badge-text {
              font-family: Inter, sans-serif;
              font-size: 0.76rem;
              font-weight: 700;
              color: var(--accent-dark);
              text-transform: uppercase;
              letter-spacing: 0.04em;
            }

            .dark-label {
              color: var(--muted);
              opacity: 0.9;
            }

            .dark-copy {
              color: var(--ink);
              opacity: 0.92;
              font-size: 0.96rem;
              line-height: 1.6;
            }

            .ai-summary-copy {
              margin: 0;
            }

            .dark-copy code {
              padding: 2px 6px;
              border-radius: 6px;
              background: rgba(24, 34, 47, 0.04);
              color: var(--accent-dark);
              font-size: 0.88em;
              font-family: monospace;
            }

            .back-btn {
              margin-top: 22px;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-weight: 700;
              font-size: 0.9rem;
              color: var(--ink);
              text-align: left;
              padding: 10px 16px;
              border-radius: 99px;
              background: rgba(24, 34, 47, 0.04);
              border: 1px solid rgba(24, 34, 47, 0.08);
              display: inline-flex;
              align-items: center;
              gap: 8px;
            }

            .back-btn:hover {
              background: rgba(24, 34, 47, 0.08);
              transform: translateX(-2px);
            }

            .results-main {
              min-width: 0;
            }

            .result-list {
              display: grid;
              gap: 20px;
            }

            .error-message {
              color: #8f1d1d;
              background: #fbe9e9;
              border: 1px solid rgba(143, 29, 29, 0.1);
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
  const [showAllActors, setShowAllActors] = useState(false);
  const [showAllTags, setShowAllTags] = useState(false);

  return (
    <motion.article
      variants={cardItemVariants}
      className="page-card movie-card"
      onClick={onSelectMovie}
      whileHover={{
        y: -3,
        borderColor: 'rgba(198, 93, 46, 0.16)',
        boxShadow: '0 16px 36px rgba(24, 34, 47, 0.04)',
      }}
    >
      <div className="movie-head">
        <span className="result-rank">{rank}</span>
        <div className="movie-heading">
          <button
            type="button"
            className="result-title-link"
            onClick={(e) => {
              e.stopPropagation();
              onSelectMovie();
            }}
          >
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
                <div className="body-copy compact-copy">
                  {showAllActors ? info.actors.join(' · ') : info.actors.slice(0, 3).join(' · ')}
                  {info.actors.length > 3 && (
                    <button
                      type="button"
                      className="actors-toggle-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        setShowAllActors(!showAllActors);
                      }}
                    >
                      {showAllActors ? ' (show less)' : ` +${info.actors.length - 3} more`}
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>

          {info.tags?.length > 0 && (
            <div className="chip-block">
              <div className="mini-label results-gap-xs">Tags</div>
              <div className="chip-row">
                {(showAllTags ? info.tags : info.tags.slice(0, 4)).map((tag) => (
                  <span key={tag} className="info-chip accent">
                    {tag}
                  </span>
                ))}
                {info.tags.length > 4 && (
                  <button
                    type="button"
                    className="info-chip neutral tag-toggle-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowAllTags(!showAllTags);
                    }}
                  >
                    {showAllTags ? 'show less' : `+${info.tags.length - 4} more`}
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      <style
        dangerouslySetInnerHTML={{
          __html: `
            .movie-card {
              padding: 28px 30px;
              cursor: pointer;
              border: 1px solid rgba(24, 34, 47, 0.04);
              box-shadow: 0 12px 32px rgba(24, 34, 47, 0.03);
            }

            .movie-head {
              display: flex;
              align-items: center;
              gap: 16px;
              margin-bottom: 20px;
            }

            .result-rank {
              width: 38px;
              height: 38px;
              border-radius: 999px;
              background: rgba(198, 93, 46, 0.08);
              color: var(--accent-dark);
              display: inline-flex;
              align-items: center;
              justify-content: center;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-weight: 800;
              font-size: 0.95rem;
              border: 1px solid rgba(198, 93, 46, 0.12);
            }

            .movie-heading {
              min-width: 0;
            }

            .result-title-link {
              padding: 0;
              color: var(--ink);
              font-size: clamp(1.5rem, 2.2vw, 2.2rem);
              line-height: 1.1;
              text-align: left;
              font-weight: 800;
              letter-spacing: -0.01em;
            }

            .result-title-link:hover {
              color: var(--accent);
            }

            .movie-grid {
              display: grid;
              grid-template-columns: 140px minmax(0, 1fr);
              gap: 28px;
            }

            .poster-column {
              display: flex;
              align-items: flex-start;
            }

            .result-poster {
              width: 100%;
              max-width: 140px;
              aspect-ratio: 2 / 3;
              object-fit: cover;
              border-radius: 16px;
              box-shadow: 0 12px 28px rgba(24, 34, 47, 0.08);
              transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            }

            .movie-card:hover .result-poster {
              transform: scale(1.03) translateY(-2px);
              box-shadow: 0 16px 36px rgba(24, 34, 47, 0.12);
            }

            .result-poster-placeholder {
              background: rgba(24, 34, 47, 0.05);
              color: var(--muted);
              display: flex;
              align-items: center;
              justify-content: center;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-size: 0.85rem;
              font-weight: 600;
              border-radius: 16px;
              aspect-ratio: 2 / 3;
              width: 100%;
              border: 1px dashed var(--line);
            }

            .movie-overview {
              margin-top: 0;
              margin-bottom: 20px;
              line-height: 1.65;
            }

            .movie-columns {
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 18px 24px;
              margin: 20px 0;
            }

            .compact-copy {
              margin: 0;
              font-size: 0.95rem;
              font-weight: 500;
            }

            .actors-toggle-btn {
              background: none;
              border: none;
              padding: 0;
              color: var(--accent);
              font-family: Inter, sans-serif;
              font-size: 0.82rem;
              font-weight: 700;
              cursor: pointer;
              margin-left: 6px;
              transition: color var(--transition-smooth);
              display: inline-block;
            }

            .actors-toggle-btn:hover {
              color: var(--accent-dark);
              text-decoration: underline;
            }

            .tag-toggle-btn {
              cursor: pointer;
              transition: all var(--transition-smooth);
            }

            .tag-toggle-btn:hover {
              background: rgba(24, 34, 47, 0.08) !important;
              border-color: rgba(24, 34, 47, 0.16) !important;
              color: var(--ink) !important;
            }

            .chip-block + .chip-block {
              margin-top: 20px;
            }

            .chip-row {
              display: flex;
              flex-wrap: wrap;
              gap: 8px;
            }

            .info-chip {
              border-radius: 999px;
              padding: 6px 12px;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-size: 0.82rem;
              font-weight: 600;
              line-height: 1.2;
              border: 1px solid transparent;
              transition: all var(--transition-smooth);
            }

            .info-chip.neutral {
              background: rgba(24, 34, 47, 0.03);
              border-color: rgba(24, 34, 47, 0.06);
              color: var(--muted);
            }

            .movie-card:hover .info-chip.neutral {
              background: #ffffff;
              border-color: rgba(24, 34, 47, 0.12);
              color: var(--ink);
            }

            .info-chip.accent {
              background: rgba(198, 93, 46, 0.08);
              border-color: rgba(198, 93, 46, 0.12);
              color: var(--accent-dark);
            }

            .movie-card:hover .info-chip.accent {
              background: rgba(198, 93, 46, 0.12);
              border-color: rgba(198, 93, 46, 0.2);
            }

            .mini-label {
              text-transform: uppercase;
              letter-spacing: 0.08em;
              font-size: 0.7rem;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-weight: 700;
              color: var(--muted);
              opacity: 0.8;
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
    </motion.article>
  );
}

export default ResultsView;
