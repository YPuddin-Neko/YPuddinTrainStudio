import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { SlidingIndicator } from '../../components/motion';
import { useWorkspaceText } from '../../utils/workspaceText';
import Projects from './Projects';
import ArchivedVersions from './ArchivedVersions';

export default function Archives() {
  const text = useWorkspaceText();
  const [params, setParams] = useSearchParams();
  const tab = params.get('tab') === 'versions' ? 'versions' : 'projects';
  const tabs = React.useRef<HTMLDivElement>(null);
  const restoreTabFocus = React.useRef(false);
  React.useLayoutEffect(() => {
    if (!restoreTabFocus.current) return;
    tabs.current?.querySelector<HTMLButtonElement>(`[data-tab="${tab}"]`)?.focus();
    restoreTabFocus.current = false;
  }, [tab]);
  const changeTab = (value: 'projects' | 'versions') => {
    if (value === tab) return;
    restoreTabFocus.current = true;
    const next = new URLSearchParams(params); next.set('tab', value); next.delete('page'); next.delete('archived');
    setParams(next);
  };
  const navigation = <div ref={tabs} className="archive-tabs ui-segmented" role="tablist" aria-label={text('归档内容', 'Archived content')} onKeyDown={event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 'projects' : event.key === 'End' ? 'versions' : tab === 'projects' ? 'versions' : 'projects';
    changeTab(next);
  }}>
    <button type="button" role="tab" data-tab="projects" aria-selected={tab === 'projects'} aria-pressed={tab === 'projects'} tabIndex={tab === 'projects' ? 0 : -1} onClick={() => changeTab('projects')}>{text('项目', 'Projects')}</button>
    <button type="button" role="tab" data-tab="versions" aria-selected={tab === 'versions'} aria-pressed={tab === 'versions'} tabIndex={tab === 'versions' ? 0 : -1} onClick={() => changeTab('versions')}>{text('版本', 'Versions')}</button>
    <SlidingIndicator className="ui-segmented-thumb"/>
  </div>;
  return tab === 'projects' ? <Projects archivedOnly archiveTabs={navigation}/> : <ArchivedVersions archiveTabs={navigation}/>;
}
