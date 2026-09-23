import StudioSelect from '../../components/StudioSelect';
import { useWorkspaceText } from '../../utils/workspaceText';

type Sources = { pypi: 'ustc' | 'tuna' | 'aliyun' | 'official'; pytorch: 'mirror' | 'aliyun' | 'sjtu' | 'official'; fallback: boolean };
const urls = { ustc: 'https://mirrors.ustc.edu.cn/pypi/simple', tuna: 'https://pypi.tuna.tsinghua.edu.cn/simple', aliyun: 'https://mirrors.aliyun.com/pypi/simple', official: 'https://pypi.org/simple' };
export default function DownloadPreferences({ value, onChange }: { value: Sources; onChange: (value: Sources) => void }) {
  const text = useWorkspaceText();
  return <section id="preferences-downloads" data-settings-section tabIndex={-1} className="settings-section">
    <div className="settings-section-heading"><div><h2>{text('软件下载源', 'Package sources')}</h2><p className="settings-note">{text('保存后用于新的软件安装。', 'Applies to new installations after saving.')}</p></div></div>
    <div className="settings-field"><label htmlFor="download-pypi">{text('Python 依赖包', 'Python packages')}</label><div className="settings-field-control">
      <StudioSelect id="download-pypi" value={value.pypi} onValueChange={pypi => onChange({ ...value, pypi: pypi as Sources['pypi'] })} options={[
        {value:'ustc',label:text('中国科学技术大学（默认）','USTC (default)')}, {value:'tuna',label:text('清华大学','Tsinghua University')}, {value:'aliyun',label:text('阿里云','Aliyun')}, {value:'official',label:text('PyPI 官方','Official PyPI')},
      ]}/><p className="settings-note break-all font-mono">{urls[value.pypi]}</p>
    </div></div>
    <div className="settings-field"><label htmlFor="download-pytorch">PyTorch</label><div className="settings-field-control">
      <StudioSelect id="download-pytorch" value={value.pytorch} onValueChange={pytorch => onChange({ ...value, pytorch: pytorch as Sources['pytorch'] })} options={[{value:'mirror',label:text('国内镜像（阿里云 → 上海交大）','Mirrors (Aliyun → SJTU)')},{value:'aliyun',label:text('阿里云','Aliyun')},{value:'sjtu',label:text('上海交通大学','Shanghai Jiao Tong University')},{value:'official',label:text('PyTorch 官方','Official PyTorch')}]} />
      <p className="settings-note">{text('macOS 使用上方的 Python 依赖包源。', 'macOS uses the Python package source above.')}</p>
    </div></div>
    <div className="settings-field"><span className="settings-field-label">{text('下载失败时', 'On download failure')}</span><div className="settings-field-control"><label className="flex items-center gap-2"><input type="checkbox" checked={value.fallback} onChange={event => onChange({ ...value, fallback: event.target.checked })}/>{text('自动尝试其他镜像和官方源', 'Try other mirrors and the official source')}</label></div></div>
  </section>;
}
