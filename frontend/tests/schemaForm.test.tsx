import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers } from '../src/mocks/handlers';
import { SchemaForm } from '../src/schema/SchemaForm/SchemaForm';
import trainSchema from '../src/schema/train-schema.json';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('SchemaForm advanced controls rendering & logic', () => {
  const sampleConfig = {
    model: { name: 'Anima-Default', dtype: 'bf16' },
    adapter: {
      algo: 'lokr',
      rank: 16,
      alpha: 16,
      factor: -1,
      rules: [],
    },
    optimizer: { lr: 0.0001, betas: [0.9, 0.999] },
    dataset: { sources: [{ path: '/data/images', repeats: 2 }] },
    sampling: { prompts: [] },
  };

  it('renders basic schema fields successfully', async () => {
    render(<SchemaForm schema={trainSchema as any} value={sampleConfig} onChange={() => {}} />);

    // 检查渲染出的顶级字段
    await waitFor(() => {
      expect(screen.getByTestId('field-adapter.algo')).toBeInTheDocument();
      expect(screen.getByTestId('field-dataset.batch_size')).toBeInTheDocument();
      expect(screen.getByTestId('field-optimizer.lr')).toBeInTheDocument();
    });
  });

  it('renders rules editor correctly with mock data', () => {
    const rulesConfig = {
      ...sampleConfig,
      adapter: {
        ...sampleConfig.adapter,
        rules: [{ match: 'blocks.*', algo: 'lokr', rank: 16, alpha: 16, factor: -1 }],
      },
    };
    render(<SchemaForm schema={trainSchema as any} value={rulesConfig} onChange={() => {}} />);

    const rulesEditor = screen.getByTestId('rules-editor');
    expect(rulesEditor).toBeInTheDocument();

    // 增加规则行
    fireEvent.click(screen.getByText('Add Rule'));
    // 增加后应有 2 个 select (algo)
    const selects = screen.getAllByRole('combobox');
    expect(selects.length).toBeGreaterThan(0);
  });

  it('renders prompts editor correctly with mock data', () => {
    const promptsConfig = {
      ...sampleConfig,
      sampling: {
        enabled: true,
        prompts: [{ prompt: 'a cat', negative_prompt: 'blur', seed: 42 }],
      },
    };
    render(<SchemaForm schema={trainSchema as any} value={promptsConfig} onChange={() => {}} />);

    expect(screen.getByTestId('prompts-editor')).toBeInTheDocument();
  });

  it('renders sources editor correctly with mock data', () => {
    render(<SchemaForm schema={trainSchema as any} value={sampleConfig} onChange={() => {}} />);

    expect(screen.getByTestId('sources-editor')).toBeInTheDocument();
    expect(screen.getByText('Add Dataset Source')).toBeInTheDocument();
  });

  it('toggles advanced fields and applies show_when logic', () => {
    const { rerender } = render(
      <SchemaForm schema={trainSchema as any} value={sampleConfig} onChange={() => {}} showAdvanced={false} />
    );

    // 在不显示高级选项时，rs_lora 不应该显示（假设它是高级）
    expect(screen.queryByText(/rs_lora/i)).not.toBeInTheDocument();

    // 打开高级选项
    rerender(
      <SchemaForm schema={trainSchema as any} value={sampleConfig} onChange={() => {}} showAdvanced={true} />
    );

    // 此时高级字段应该出现
    expect(screen.getByTestId('field-adapter.rs_lora')).toBeInTheDocument();

    // 修改 algo 为 'lora'，factor 的 show_when 应隐藏该字段
    const loraConfig = { ...sampleConfig, adapter: { ...sampleConfig.adapter, algo: 'lora' } };
    rerender(
      <SchemaForm schema={trainSchema as any} value={loraConfig} onChange={() => {}} showAdvanced={true} />
    );

    expect(screen.queryByTestId('field-adapter.factor')).not.toBeInTheDocument();
  });
});
