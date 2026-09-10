import React from 'react';
import { apiClient } from '../api/client';
import { FsListResponse } from '../api/types';
import { FolderOpen } from 'lucide-react';

export const PathPickerModal: React.FC<{
  isOpen: boolean;
  initialPath?: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}> = ({ isOpen, initialPath = '/', onSelect, onClose }) => {
  const [currentPath, setCurrentPath] = React.useState(initialPath);
  const [data, setData] = React.useState<FsListResponse | null>(null);

  React.useEffect(() => {
    if (isOpen) {
      setCurrentPath(initialPath || '/');
    }
  }, [isOpen, initialPath]);

  React.useEffect(() => {
    if (isOpen) {
      apiClient
        .get<FsListResponse>('/fs/list', { params: { path: currentPath } })
        .then(setData)
        .catch(console.error);
    }
  }, [isOpen, currentPath]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div
        className="bg-white dark:bg-slate-800 rounded-xl max-w-lg w-full p-6 space-y-4 shadow-xl border border-slate-200 dark:border-slate-700"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center border-b pb-2 dark:border-slate-700">
          <h3 className="font-semibold text-lg">Browse Server Path</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">✕</button>
        </div>
        <div className="text-xs font-mono bg-slate-100 dark:bg-slate-900 p-2 rounded truncate">
          Current: {data?.path || currentPath}
        </div>
        <div className="max-h-60 overflow-y-auto divide-y divide-slate-100 dark:divide-slate-700">
          {data?.parent && (
            <div
              onClick={() => setCurrentPath(data.parent!)}
              className="p-2 text-sm hover:bg-slate-50 dark:hover:bg-slate-700 cursor-pointer font-medium text-blue-500"
            >
              📁 .. (Parent Directory)
            </div>
          )}
          {(data?.entries ?? []).map((entry) => (
            <div
              key={entry.name}
              onClick={() => {
                if (entry.is_dir) {
                  setCurrentPath(`${data!.path === '/' ? '' : data!.path}/${entry.name}`);
                } else {
                  onSelect(`${data!.path === '/' ? '' : data!.path}/${entry.name}`);
                  onClose();
                }
              }}
              className="p-2 text-sm hover:bg-slate-50 dark:hover:bg-slate-700 flex justify-between items-center cursor-pointer"
            >
              <span>{entry.is_dir ? '📁' : '📄'} {entry.name}</span>
              <span className="text-xs text-slate-400">{entry.is_dir ? 'dir' : `${entry.size} B`}</span>
            </div>
          ))}
        </div>
        <div className="flex justify-end space-x-2 pt-2 border-t dark:border-slate-700">
          <button onClick={onClose} className="px-3 py-1.5 text-sm rounded bg-slate-200 dark:bg-slate-700">Cancel</button>
          <button
            onClick={() => {
              onSelect(data?.path || currentPath);
              onClose();
            }}
            className="px-3 py-1.5 text-sm rounded bg-blue-600 text-white"
          >
            Select Current Dir
          </button>
        </div>
      </div>
    </div>
  );
};

export const PathInput: React.FC<{
  value: string;
  onChange: (val: string) => void;
  placeholder?: string;
}> = ({ value = '', onChange, placeholder }) => {
  const [modalOpen, setModalOpen] = React.useState(false);

  return (
    <div className="flex space-x-2">
      <input
        type="text"
        value={value || ''}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="flex-1 px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600 font-mono"
      />
      <button
        type="button"
        onClick={() => setModalOpen(true)}
        className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-md hover:bg-slate-300 dark:hover:bg-slate-600 text-sm flex items-center space-x-1"
      >
        <FolderOpen className="w-4 h-4" />
        <span>Browse</span>
      </button>
      <PathPickerModal
        isOpen={modalOpen}
        initialPath={value || '/'}
        onSelect={onChange}
        onClose={() => setModalOpen(false)}
      />
    </div>
  );
};
