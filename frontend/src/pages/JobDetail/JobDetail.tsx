import { useParams } from 'react-router-dom';

export default function JobDetail() {
  const { id } = useParams<{ id: string }>();

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Job Detail</h2>
      <p className="text-slate-500">Viewing detail for Job ID: {id}</p>
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 h-64 flex items-center justify-center text-slate-500">
        Charts and metrics will render here (M3).
      </div>
    </div>
  );
}
