import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import ParameterSections from '../../../frontend/src/components/ParameterSections';
import { presentConfigIssues } from '../../../frontend/src/utils/configPresentation';
import '../../../frontend/src/i18n';

afterEach(() => vi.restoreAllMocks());
it('shows current validation status without treating an unchecked model path as complete', () => {
  const root = document.createElement('div');
  root.innerHTML = '<section data-group="model"><button class="config-group-title">Model</button></section>';
  const props = {rootRef:{current:root}, tab:'model' as const, onTabChange:vi.fn(), onRevealAdvanced:vi.fn()};
  const view = render(<ParameterSections {...props}/>);
  const model = screen.getByRole('button', {name:/模型选择$/});
  const optimizer = screen.getByRole('button', {name:/优化器$/});
  expect(model).toHaveAccessibleDescription('待检查');
  expect(optimizer).toHaveAccessibleDescription('待检查');
  const issues = presentConfigIssues([{loc:'model.dit_path',msg:'Weight file is missing'}]);
  view.rerender(<ParameterSections {...props} checked issues={issues}/>);
  expect(model).toHaveAccessibleDescription('1 项待配置');
  expect(optimizer).toHaveAccessibleDescription('检查通过');
  // Schema-only success cannot prove that a local model exists.
  view.rerender(<ParameterSections {...props} checked/>);
  expect(model).toHaveAccessibleDescription('待检查');
  view.rerender(<ParameterSections {...props} checked planChecked/>);
  expect(model).toHaveAccessibleDescription('检查通过');
  expect(model).toHaveAttribute('aria-current','step');
  expect(model.querySelector('.parameter-step-complete')).not.toBeNull();
  // Edits and requests in progress invalidate the old success state.
  view.rerender(<ParameterSections {...props}/>);
  expect(model).toHaveAccessibleDescription('待检查');
  expect(optimizer).toHaveAccessibleDescription('待检查');
  view.rerender(<ParameterSections {...props} checked planChecked issues={presentConfigIssues([{loc:'device',msg:'unavailable'}])}/>);
  expect(model).toHaveAccessibleDescription('待检查');
});
it('keeps a short final section selected after a clamped jump, then follows user scrolling', async () => {
  vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => { callback(0); return 1; });
  const root = document.createElement('div');
  root.innerHTML = '<section data-group="objective"><button class="config-group-title" aria-expanded="true">Noise</button></section><section data-group="logging"><button class="config-group-title" aria-expanded="true">Logging</button></section>';
  document.body.append(root);
  let offset = 0;
  Object.defineProperties(root, { scrollTop: { get: () => offset, set: (value: number) => { offset = Math.max(0, Math.min(100, value)); } }, scrollHeight: { value: 700 }, clientHeight: { value: 600 } });
  vi.spyOn(root, 'getBoundingClientRect').mockImplementation(() => ({ top: 200 }) as DOMRect);
  vi.spyOn(root.children[0], 'getBoundingClientRect').mockImplementation(() => ({ top: 200 - offset }) as DOMRect);
  vi.spyOn(root.children[1], 'getBoundingClientRect').mockImplementation(() => ({ top: 600 - offset }) as DOMRect);
  render(<ParameterSections rootRef={{current:root}} tab="advanced" onTabChange={() => {}} onRevealAdvanced={() => {}}/>);
  const logging = screen.getByRole('button', { name: /训练记录$/ });
  fireEvent.click(logging);
  expect(offset).toBe(100);
  fireEvent.scroll(root);
  root.append(document.createElement('span'));
  await act(async () => {});
  expect(logging).toHaveAttribute('aria-current','step');
  root.scrollTop = 0; fireEvent.scroll(root);
  expect(screen.getByRole('button',{name:/噪声与损失$/})).toHaveAttribute('aria-current','step');
  root.scrollTop = 100; fireEvent.scroll(root);
  expect(logging).toHaveAttribute('aria-current','step');
  root.remove();
});

it('maps old training links and component errors to the model section', () => {
  vi.spyOn(window, 'requestAnimationFrame').mockImplementation(callback => { callback(0); return 1; });
  const root = document.createElement('div');
  root.innerHTML = '<section data-group="model"><button class="config-group-title" aria-expanded="true">Model</button></section>';
  const reveal = vi.fn();
  render(<ParameterSections rootRef={{current:root}} tab="train" group="training" hasTrainingMode
    onTabChange={vi.fn()} onRevealAdvanced={reveal} checked planChecked
    issues={presentConfigIssues([{loc:'training.train_text_encoder',msg:'Unsupported'}])}/>);
  const model = screen.getByRole('button', {name:/模型选择$/});
  expect(model).toHaveAttribute('aria-current', 'step');
  expect(model).toHaveAccessibleDescription('1 项待配置');
  expect(screen.queryByRole('button', {name:/训练方式$/})).not.toBeInTheDocument();
  expect(screen.getByRole('button', {name:/优化器$/})).toHaveAccessibleDescription('检查通过');
  expect(reveal).not.toHaveBeenCalled();
});
