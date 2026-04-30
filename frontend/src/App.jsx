import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import Search from './components/Search';
import ProjectOverview from './components/ProjectOverview';
import ResultsView from './components/ResultsView';
import DetailView from './components/DetailView';
import './index.css';

function normalizeUseAgent(value) {
  return value === '1' || value === 'true' || value === true;
}

function parseRoute(locationLike = window.location) {
  const params = new URLSearchParams(locationLike.search);
  const pathname = locationLike.pathname || '/';
  const useAgent = normalizeUseAgent(params.get('use_agent'));
  const text = params.get('text') || '';
  const title = params.get('title') || '';

  if (pathname === '/movie') {
    return {
      kind: 'detail',
      pathname,
      title,
      useAgent,
    };
  }

  if (pathname === '/about') {
    return {
      kind: 'about',
      pathname,
      text: '',
      useAgent: true,
    };
  }

  if (pathname === '/search' || pathname === '/search/results') {
    return {
      kind: pathname === '/search/results' ? 'results' : 'search',
      pathname,
      text,
      useAgent,
    };
  }

  return {
    kind: 'search',
    pathname,
    text: '',
    useAgent: true,
  };
}

function buildSearchHref(text, useAgent) {
  const params = new URLSearchParams();
  params.set('text', text);
  params.set('use_agent', useAgent ? '1' : '0');
  return `/search/results?${params.toString()}`;
}

function buildMovieHref(title, useAgent) {
  const params = new URLSearchParams();
  params.set('title', title);
  params.set('use_agent', useAgent ? '1' : '0');
  return `/movie?${params.toString()}`;
}

function App() {
  const [route, setRoute] = useState(() => ({ ...parseRoute(), navToken: 0 }));
  const [config, setConfig] = useState(null);

  useEffect(() => {
    let cancelled = false;

    axios
      .get('/api/config')
      .then((response) => {
        if (!cancelled) {
          setConfig(response.data);
        }
      })
      .catch((error) => {
        console.error('Failed to load app config.', error);
      });

    const handlePopState = () => setRoute({ ...parseRoute(), navToken: Date.now() });
    window.addEventListener('popstate', handlePopState);

    return () => {
      cancelled = true;
      window.removeEventListener('popstate', handlePopState);
    };
  }, []);

  const defaultQuery = config?.default_query || 'Mind-bending movies like Inception but darker';
  const agentAvailable = config?.agent?.available ?? true;
  const appYear = useMemo(() => new Date().getFullYear(), []);

  const hardNavigate = (href) => {
    window.location.assign(href);
  };

  const goAbout = () => hardNavigate('/about');
  const goSearch = () => hardNavigate('/search');

  const startSearch = ({ text, useAgent = true }) => {
    const normalized = (text || '').trim() || defaultQuery;
    hardNavigate(buildSearchHref(normalized, useAgent));
  };

  const openMovie = ({ title, useAgent }) => {
    hardNavigate(buildMovieHref(title, useAgent));
  };

  const goBackFromDetail = () => {
    if (window.history.length > 1) {
      window.history.back();
      return;
    }
    hardNavigate('/search');
  };

  return (
    <div className="app-shell">
      <Navbar
        currentView={route.kind}
        onSearch={goSearch}
        onAbout={goAbout}
      />

      <main>
        {route.kind === 'search' && (
          <Search
            key="search"
            defaultQuery={defaultQuery}
            agentAvailable={agentAvailable}
            onSearch={startSearch}
          />
        )}

        {route.kind === 'about' && <ProjectOverview key="about" />}

        {route.kind === 'results' && (
          <ResultsView
            key={`results:${route.text}:${route.useAgent ? '1' : '0'}:${route.navToken}`}
            query={route.text}
            useAgent={route.useAgent}
            defaultQuery={defaultQuery}
            agentAvailable={agentAvailable}
            onSearch={startSearch}
            onSelectMovie={openMovie}
          />
        )}

        {route.kind === 'detail' && route.title && (
          <DetailView
            key={`movie:${route.title}:${route.useAgent ? '1' : '0'}:${route.navToken}`}
            title={route.title}
            useAgent={route.useAgent}
            onBack={goBackFromDetail}
            onSelectMovie={openMovie}
          />
        )}
      </main>

      <footer className="app-footer">
        <p className="body-copy">
          CineSeek Retrieval System &copy; {appYear} &bull; Built with FastAPI, FAISS, MCP, and React
        </p>
      </footer>

      <style
        dangerouslySetInnerHTML={{
          __html: `
            .app-footer {
              margin-top: 60px;
              padding-top: 24px;
              border-top: 1px solid var(--line);
              text-align: center;
            }

            .app-footer p {
              font-size: 0.85rem;
              opacity: 0.72;
            }
          `,
        }}
      />
    </div>
  );
}

export default App;
