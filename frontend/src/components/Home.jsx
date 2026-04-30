import { Search } from 'lucide-react';
import { useRef, useState } from 'react';

const HOME_EXAMPLES = [
  'Mind-bending movies like Inception but darker',
  'Emotional sci-fi about space, family, and sacrifice',
  'Underrated psychological thrillers',
];

function Home({ defaultQuery, agentAvailable, onSearch }) {
  const textareaRef = useRef(null);
  const [loading, setLoading] = useState(false);

  const submitSearch = (event) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const text = String(formData.get('text') || '');
    const useAgent = agentAvailable ? formData.get('use_agent') === '1' : false;
    setLoading(true);
    onSearch({ text, useAgent });
  };

  const fillQuery = (query) => {
    if (!textareaRef.current) {
      return;
    }
    textareaRef.current.value = query;
    textareaRef.current.focus();
  };

  return (
    <div className="home-view">
      <section className="page-card hero-card">
        <div className="eyebrow home-gap-sm">Search Interface</div>
        <h1 className="display-title hero-title">CineSeek — LLM-powered semantic movie search.</h1>
        <p className="body-copy hero-lede">
          Search movies using natural language, not keywords. CineSeek combines dense retrieval, FAISS candidate
          search, and LLM reranking to better match user intent on fuzzy movie queries.
        </p>

        <form className="search-stack home-gap-md" onSubmit={submitSearch}>
          <label className="mini-label home-gap-xs" htmlFor="home-query-box">
            Try a query
          </label>
          <textarea
            id="home-query-box"
            name="text"
            className="query-box"
            placeholder={`Try: "${defaultQuery}"`}
            rows={4}
            ref={textareaRef}
          />

          <div className="hero-actions">
            <div>
              <label className="toggle-row">
                <input type="checkbox" name="use_agent" value="1" defaultChecked={agentAvailable} disabled={!agentAvailable} />
                <span>Use LLM agent for reranking and explanation</span>
              </label>
              <div className="mini-label hero-note">Results are powered by semantic retrieval + LLM reranking.</div>
            </div>
            <button type="submit" className="primary-btn">
              <Search size={18} />
              <span>Search Now</span>
            </button>
          </div>

          <div className="home-gap-md">
            <div className="mini-label home-gap-xs">Try one of these</div>
            <div className="chip-row">
              {HOME_EXAMPLES.map((query) => (
                <button
                  key={query}
                  type="button"
                  className="query-chip"
                  onClick={() => fillQuery(query)}
                >
                  {query}
                </button>
              ))}
            </div>
          </div>

          {loading && (
            <div className="loading-panel">
              <div className="loading-row">
                <div className="loading-orb" aria-hidden="true"></div>
                <div>
                  <div className="mini-label home-gap-xs">Working</div>
                  <div className="body-copy loading-copy">
                    Please wait while CineSeek retrieves candidates, reranks them, and prepares the final results.
                  </div>
                </div>
              </div>
            </div>
          )}
        </form>
      </section>

      <style
        dangerouslySetInnerHTML={{
          __html: `
            .home-view {
              display: flex;
              flex-direction: column;
              gap: 20px;
            }

            .hero-card {
              padding: 32px;
            }

            .hero-title {
              max-width: none;
            }

            .hero-lede {
              max-width: 82ch;
              font-size: 1.12rem;
            }

            .home-gap-xs {
              margin-bottom: 8px;
            }

            .home-gap-sm {
              margin-bottom: 12px;
            }

            .home-gap-md {
              margin-top: 20px;
            }

            .search-stack {
              display: flex;
              flex-direction: column;
            }

            .query-box {
              width: 100%;
              min-height: 128px;
              border-radius: 22px;
              border: 1px solid var(--line);
              background: rgba(255, 255, 255, 0.92);
              padding: 18px 20px;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-size: 1rem;
              line-height: 1.6;
              box-shadow: inset 0 1px 0 rgba(24, 34, 47, 0.04);
              resize: vertical;
            }

            .query-box:focus {
              outline: none;
              box-shadow: 0 0 0 0.25rem rgba(198, 93, 46, 0.16);
              border-color: rgba(198, 93, 46, 0.45);
            }

            .hero-actions {
              display: flex;
              align-items: flex-start;
              justify-content: space-between;
              gap: 16px;
              margin-top: 16px;
            }

            .toggle-row {
              display: inline-flex;
              align-items: center;
              gap: 10px;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              color: var(--muted);
            }

            .toggle-row input {
              margin: 0;
            }

            .hero-note {
              margin-top: 8px;
            }

            .primary-btn {
              display: inline-flex;
              align-items: center;
              justify-content: center;
              gap: 10px;
              background: linear-gradient(135deg, var(--accent), #e77d4e);
              color: #fff;
              border-radius: 999px;
              padding: 13px 22px;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-weight: 700;
              box-shadow: 0 12px 30px rgba(198, 93, 46, 0.22);
              transition: transform 0.18s ease, box-shadow 0.18s ease;
            }

            .primary-btn:hover {
              transform: translateY(-1px);
              box-shadow: 0 16px 32px rgba(198, 93, 46, 0.28);
            }

            .chip-row {
              display: flex;
              flex-wrap: wrap;
              gap: 10px;
            }

            .loading-panel {
              margin-top: 20px;
              border-radius: 24px;
              border: 1px solid var(--line);
              background: rgba(255, 255, 255, 0.72);
              padding: 20px;
            }

            .loading-row {
              display: flex;
              align-items: center;
              gap: 14px;
            }

            .loading-orb {
              width: 1.1rem;
              height: auto;
              aspect-ratio: 1 / 1;
              border-radius: 999px;
              border: 2px solid rgba(198, 93, 46, 0.22);
              border-top-color: var(--accent);
              animation: spin 1s linear infinite;
              flex: 0 0 auto;
            }

            .loading-copy {
              margin: 0;
            }

            .query-chip {
              border: 1px solid rgba(24, 34, 47, 0.12);
              border-radius: 999px;
              background: rgba(255, 255, 255, 0.78);
              color: var(--ink);
              padding: 10px 14px;
              text-align: left;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-size: 0.92rem;
              transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
            }

            .query-chip:hover {
              background: rgba(198, 93, 46, 0.08);
              border-color: rgba(198, 93, 46, 0.24);
              transform: translateY(-1px);
            }

            @media (max-width: 720px) {
              .hero-actions {
                flex-direction: column;
              }

              .primary-btn {
                width: 100%;
              }
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

export default Home;
