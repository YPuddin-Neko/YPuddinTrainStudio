import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import SampleLoss from '../src/components/SampleLoss';
import i18n from '../src/i18n';

beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });

describe('sample training loss', () => {
  it('keeps a real zero and rounds long values for the gallery', () => {
    const { rerender } = render(<SampleLoss sample={{ step: 10, loss: 0 }} />);
    expect(screen.getByText('0')).toHaveAttribute('data-recorded', 'true');
    rerender(<SampleLoss sample={{ step: 10, loss: 1.472497820854187 }} />);
    expect(screen.getByText('1.4725')).toBeInTheDocument();
    expect(screen.getByTitle(/第 10 步记录的训练损失/)).toHaveTextContent('训练损失');
  });

  it('distinguishes an initial sample from missing or invalid training records', () => {
    const { rerender } = render(<SampleLoss sample={{ step: 0, loss: null }} />);
    expect(screen.getByText('初始采样 · 未训练')).toBeInTheDocument();
    for (const loss of [null, undefined, Number.NaN, Number.POSITIVE_INFINITY]) {
      rerender(<SampleLoss sample={{ step: 10, loss }} />);
      expect(screen.getByText('未记录')).toHaveAttribute('data-recorded', 'false');
    }
  });
});
