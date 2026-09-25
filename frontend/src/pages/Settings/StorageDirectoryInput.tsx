import React from 'react';
import { PathInput } from '../../components/PathBrowser';
import StudioSelect from '../../components/StudioSelect';
import { useWorkspaceText } from '../../utils/workspaceText';

export type StoragePathPreview = { path: string; browse_root: string };

export default function StorageDirectoryInput({ label, value, preview, resolveDefaultPath, onChange }: {
  label: string;
  value: string;
  preview?: StoragePathPreview;
  resolveDefaultPath: () => Promise<string>;
  onChange: (value: string) => void;
}) {
  const text = useWorkspaceText();
  const [editing, setEditing] = React.useState(false);
  const custom = !!value || editing;
  const displayedPath = preview?.path
    .replace('{project_id}', text('{项目}', '{project}'))
    .replace('{version}', text('{版本}', '{version}'))
    .replace('{job_id}', text('{任务}', '{job}')) || '';
  return <div className="storage-directory-control">
    <StudioSelect aria-label={`${label} · ${text('位置方式', 'Location mode')}`} value={custom ? 'custom' : 'default'}
      options={[{value:'default',label:text('默认目录','Default')},{value:'custom',label:text('自定义','Custom')}]}
      onValueChange={mode => { setEditing(mode === 'custom'); if (mode === 'default') onChange(''); }}/>
    <PathInput ariaLabel={label} value={custom ? value : displayedPath} readOnly={!custom}
      browsePath={custom ? undefined : preview?.browse_root || ''} resolveDefaultPath={resolveDefaultPath}
      directoryOnly allowMissingDirectory placeholder={custom ? text('输入目录路径','Enter a directory path') : ''}
      onChange={path => { setEditing(true); onChange(path); }}/>
  </div>;
}
