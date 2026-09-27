import React from 'react';
import { Link } from 'react-router-dom';
import TopbarBreadcrumb from './TopbarBreadcrumb';
import { useWorkspaceText } from '../utils/workspaceText';

/** A top-level page's place in the top bar: its name, after the section it belongs to when there is one. */
export default function PageLocation({ trail }: { trail: Array<{ label: string; to?: string }> }) {
  const text = useWorkspaceText();
  return <TopbarBreadcrumb topbarOnly>
    <nav className="workspace-breadcrumb" aria-label={text('当前位置', 'Current location')}>
      {trail.map((item, index) => <React.Fragment key={item.label}>
        {index > 0 && <span aria-hidden="true">/</span>}
        {item.to ? <Link to={item.to}>{item.label}</Link> : <span>{item.label}</span>}
      </React.Fragment>)}
    </nav>
  </TopbarBreadcrumb>;
}
