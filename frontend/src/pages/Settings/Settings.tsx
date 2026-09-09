import React from 'react';
import { apiClient } from '../../api/client';
import { Settings as SettingsType } from '../../api/types';

export default function Settings() {
  const [settings, setSettings] = React.useState<SettingsType | null>(null);

  React.useEffect(() => {
    apiClient.get<SettingsType>('/settings').then(setSettings).catch(console.error);
  }, []);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Settings</h2>
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 space-y-4">
        <div>
          <label className="text-sm font-medium">Data Root</label>
          <input
            type="text"
            className="w-full mt-1 border border-slate-300 dark:border-slate-600 rounded-md p-2 dark:bg-slate-900"
            value={settings?.paths?.data_root || ''}
            disabled
          />
        </div>
      </div>
    </div>
  );
}
