import type { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { KeyRound, RefreshCw } from 'lucide-react';
import { useWorkspaceText } from '../../utils/workspaceText';

export function ModelToolbar({ embedded, directory, credentialsLink, refreshing = false, onRefresh, children }: {
  embedded: boolean;
  directory?: string;
  credentialsLink: string;
  refreshing?: boolean;
  onRefresh: () => unknown;
  children: ReactNode;
}) {
  const text = useWorkspaceText(), location = useLocation();
  return <div className="models-toolbar">
    <div className="models-heading">
      <div><h2>{text('模型权重', 'Model weights')}</h2>{!embedded && <p>{text('准备模型组件，供项目选择。', 'Prepare components for your projects.')}</p>}</div>
      {directory && <div className="models-heading-path" title={directory}><span>{text('模型目录', 'Model directory')}</span><strong>{directory}</strong><Link className="ui-link" to="/settings/preferences?section=storage" replace state={location.state}>{text('更改', 'Change')}</Link></div>}
      <div className="model-actions"><Link to={credentialsLink} replace state={location.state} className="ui-btn"><KeyRound size={14}/>{text('访问密钥', 'Access keys')}</Link><button type="button" className="ui-btn ui-btn-icon" disabled={refreshing} onClick={() => void onRefresh()} aria-label={text('刷新模型', 'Refresh models')} title={text('刷新模型', 'Refresh models')}><RefreshCw size={14}/></button></div>
    </div>
    <div className="models-filters">{children}</div>
  </div>;
}

export function ModelCatalogSection({ heading, action, label, testId, children }: {
  heading?: ReactNode;
  action?: ReactNode;
  label?: string;
  testId?: string;
  children: ReactNode;
}) {
  return <section className="model-component" aria-label={label} data-testid={testId}>
    {(heading || action) && <header>{heading && <h3>{heading}</h3>}{action}</header>}
    {children}
  </section>;
}

export function ModelCatalogRow({ name, nameTitle, metadata, action, testId, className = '', children }: {
  name: string;
  nameTitle?: string;
  metadata: ReactNode;
  action: ReactNode;
  testId?: string;
  className?: string;
  children?: ReactNode;
}) {
  return <div className={`model-catalog-row ${className}`.trim()} data-testid={testId}>
    <div className="model-catalog-description"><strong title={nameTitle}>{name}</strong><div>{metadata}</div>{children}</div>
    <div className="model-catalog-action">{action}</div>
  </div>;
}
