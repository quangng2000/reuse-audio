import { useLiveDemo } from './controller';
import { Console } from './view/Console';
import { Introduction, ResearchNotice } from './view/Introduction';
import { Playback } from './view/Playback';

export function App() {
  const demo = useLiveDemo();
  return <main>
    <header><a className="wordmark" href="https://github.com/quangng2000/reuse-audio">RE-USE <span>audio</span></a>
      <span className="badge">Browser research preview</span></header>
    <Introduction /><Console demo={demo} /><Playback result={demo.result} /><ResearchNotice />
  </main>;
}
