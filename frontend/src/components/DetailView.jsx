import { useEffect, useState } from 'react';
import axios from 'axios';
import { ArrowLeft, Film, Loader2, User } from 'lucide-react';
import { motion } from 'framer-motion';

function DetailView({ title, useAgent, onBack, onSelectMovie }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    axios
      .get('/api/movie', {
        params: { title, use_agent: useAgent },
      })
      .then((response) => {
        if (!cancelled) {
          setData(response.data);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError('Failed to load movie details.');
          console.error(err);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [title, useAgent]);

  if (loading) {
    return (
      <div className="loading-state page-card">
        <Loader2 className="spinner" size={42} />
        <p className="body-copy loading-copy">Fetching movie metadata and similar picks...</p>

        <style
          dangerouslySetInnerHTML={{
            __html: `
              .loading-state {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 14px;
                min-height: 280px;
              }

              .loading-copy {
                margin: 0;
              }

              .spinner {
                color: var(--accent);
                animation: spin 1s linear infinite;
              }

              @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
              }
            `,
          }}
        />
      </div>
    );
  }

  if (error || !data) {
    return <div className="error-message">{error || 'Movie not found.'}</div>;
  }

  const { movie, related } = data;
  const info = movie.structured || {};

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="detail-view">
      <button type="button" className="nav-pill detail-back" onClick={onBack}>
        <ArrowLeft size={18} />
        <span>Back</span>
      </button>

      <section className="page-card detail-hero">
        <div className="detail-grid">
          <div className="poster-stack">
            <div className="eyebrow detail-gap-sm">Movie Detail</div>
            {movie.poster_url ? (
              <img src={movie.poster_url} alt={`Poster for ${movie.title}`} className="detail-poster" loading="lazy" />
            ) : (
              <div className="detail-poster detail-placeholder">No Poster</div>
            )}
          </div>

          <div className="info-stack">
            <div className="eyebrow detail-gap-sm">{info.release_year || 'Unknown Year'}</div>
            <h1 className="display-title detail-title">{movie.title}</h1>

            {info.genres?.length > 0 && (
              <div className="chip-row detail-gap-md">
                {info.genres.map((genre) => (
                  <span key={genre} className="info-chip neutral">
                    {genre}
                  </span>
                ))}
              </div>
            )}

            {info.overview && <p className="body-copy detail-copy">{info.overview}</p>}

            <div className="credits-grid">
              {info.director && (
                <div className="credit-card">
                  <div className="credit-head">
                    <User size={16} />
                    <span className="mini-label">Director</span>
                  </div>
                  <div className="body-copy compact-copy">{info.director}</div>
                </div>
              )}

              {info.actors?.length > 0 && (
                <div className="credit-card">
                  <div className="credit-head">
                    <Film size={16} />
                    <span className="mini-label">Actors</span>
                  </div>
                  <div className="body-copy compact-copy">{info.actors.join(', ')}</div>
                </div>
              )}
            </div>

            {info.tags?.length > 0 && (
              <div className="detail-gap-md">
                <div className="mini-label detail-gap-xs">Tags</div>
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
      </section>

      <section className="detail-gap-lg">
        <div className="eyebrow detail-gap-sm">Similar Picks</div>
        <div className="related-grid">
          {related.map((candidate, index) => (
            <button
              key={`${candidate.title}-${index}`}
              type="button"
              className="page-card related-card"
              onClick={() => onSelectMovie({ title: candidate.title, useAgent })}
            >
              <div className="related-card-grid">
                {candidate.poster_url ? (
                  <img
                    src={candidate.poster_url}
                    alt={`Poster for ${candidate.title}`}
                    className="related-poster"
                    loading="lazy"
                  />
                ) : (
                  <div className="related-poster detail-placeholder">No Poster</div>
                )}

                <div className="related-copy">
                  <h2 className="related-title">{candidate.title}</h2>
                  {candidate.structured?.overview && (
                    <p className="body-copy compact-copy">{candidate.structured.overview}</p>
                  )}
                </div>
              </div>
            </button>
          ))}
        </div>
      </section>

      <style
        dangerouslySetInnerHTML={{
          __html: `
            .detail-view {
              display: flex;
              flex-direction: column;
              gap: 20px;
            }

            .detail-back {
              display: inline-flex;
              align-items: center;
              gap: 8px;
              width: fit-content;
            }

            .detail-hero {
              padding: 30px;
            }

            .detail-grid {
              display: grid;
              grid-template-columns: 250px minmax(0, 1fr);
              gap: 28px;
            }

            .detail-title {
              font-size: clamp(2rem, 4vw, 3.4rem);
              margin-bottom: 0;
            }

            .detail-poster {
              width: 100%;
              aspect-ratio: 2 / 3;
              object-fit: cover;
              border-radius: 22px;
              box-shadow: 0 22px 44px rgba(24, 34, 47, 0.16);
            }

            .detail-placeholder {
              background: rgba(24, 34, 47, 0.08);
              color: var(--muted);
              display: flex;
              align-items: center;
              justify-content: center;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
            }

            .detail-gap-xs {
              margin-bottom: 8px;
            }

            .detail-gap-sm {
              margin-bottom: 12px;
            }

            .detail-gap-md {
              margin-top: 20px;
            }

            .detail-gap-lg {
              margin-top: 8px;
            }

            .detail-copy {
              margin: 18px 0 0;
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

            .credits-grid {
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 16px;
              margin-top: 24px;
            }

            .credit-card {
              background: rgba(255, 255, 255, 0.62);
              border: 1px solid var(--line);
              border-radius: 18px;
              padding: 16px;
            }

            .credit-head {
              display: flex;
              align-items: center;
              gap: 8px;
              margin-bottom: 8px;
              color: var(--accent-dark);
            }

            .compact-copy {
              margin: 0;
            }

            .related-grid {
              display: grid;
              gap: 16px;
            }

            .related-card {
              text-align: left;
              padding: 18px;
              transition: transform 0.18s ease, border-color 0.18s ease;
            }

            .related-card:hover {
              transform: translateY(-1px);
              border-color: rgba(198, 93, 46, 0.24);
            }

            .related-card-grid {
              display: grid;
              grid-template-columns: 120px minmax(0, 1fr);
              gap: 18px;
              align-items: start;
            }

            .related-poster {
              width: 100%;
              aspect-ratio: 2 / 3;
              object-fit: cover;
              border-radius: 16px;
            }

            .related-title {
              margin: 0 0 8px;
              font-size: 1.5rem;
              line-height: 1.1;
            }

            @media (max-width: 860px) {
              .detail-grid,
              .credits-grid,
              .related-card-grid {
                grid-template-columns: 1fr;
              }

              .poster-stack {
                max-width: 240px;
              }
            }
          `,
        }}
      />
    </motion.div>
  );
}

export default DetailView;
