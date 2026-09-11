import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import AccessKeys, { type CredentialProvider, type CredentialStates } from '../src/pages/Settings/AccessKeys';
import i18n from '../src/i18n';

const server = setupServer();
let configured: CredentialStates;
let save = vi.fn();
let clear = vi.fn();
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
afterEach(() => { cleanup(); server.resetHandlers(); vi.restoreAllMocks(); });
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  configured = { huggingface: { configured: false }, modelscope: { configured: false }, danbooru: { configured: false }, gelbooru: { configured: false } };
  save=vi.fn();clear=vi.fn();
  server.use(
    http.get('/api/credentials', () => HttpResponse.json(configured)),
    http.put('/api/credentials/:provider', async ({ params, request }) => {
      const provider=params.provider as CredentialProvider;
      save(provider,await request.json()); configured[provider]={configured:true};
      return HttpResponse.json(configured[provider]);
    }),
    http.delete('/api/credentials/:provider', ({ params }) => {
      const provider=params.provider as CredentialProvider;
      clear(provider); configured[provider]={configured:false};
      return HttpResponse.json(configured[provider]);
    }),
  );
});
const mount = () => render(<MemoryRouter initialEntries={['/settings/environment?tab=credentials']}><AccessKeys/></MemoryRouter>);

describe('central access keys',()=>{
  it.each([
    ['huggingface','Hugging Face 访问令牌',undefined, {token:'fake-secret'}],
    ['modelscope','ModelScope 访问令牌',undefined, {token:'fake-secret'}],
    ['danbooru','Danbooru API Key','Danbooru 用户名',{username:'test_user',api_key:'fake-secret'}],
    ['gelbooru','Gelbooru API Key','Gelbooru 用户 ID',{user_id:'1234',api_key:'fake-secret'}],
  ] as const)('saves and clears %s without displaying saved values',async(provider,label,accountLabel,payload)=>{
    configured[provider]={configured:true};
    const changed=vi.fn();window.addEventListener('credentials.changed',changed);
    mount();
    const input=screen.getByLabelText(label);const form=input.closest('form')!;
    await waitFor(()=>expect(within(form).getByRole('button',{name:'清除'})).toBeEnabled());
    expect(input).toHaveAttribute('type','password');expect(input).toHaveValue('');
    expect(form.querySelector(`label[for="token-${provider}"]`)).toHaveTextContent(accountLabel ? 'API Key' : '访问令牌');
    expect(form.querySelector(`label[for="token-${provider}"]`)).not.toHaveTextContent(provider === 'huggingface' ? 'Hugging Face' : provider === 'modelscope' ? 'ModelScope' : provider === 'danbooru' ? 'Danbooru' : 'Gelbooru');
    if(accountLabel){const account=screen.getByLabelText(accountLabel);expect(account).toHaveValue('');fireEvent.change(account,{target:{value:provider==='gelbooru'?'1234':'test_user'}});}
    fireEvent.change(input,{target:{value:'fake-secret'}});
    fireEvent.click(within(form).getByRole('button',{name:'保存'}));
    await waitFor(()=>expect(save).toHaveBeenCalledWith(provider,payload));
    await waitFor(()=>expect(input).toHaveValue(''));
    expect(within(form).getByRole('status')).toHaveTextContent('已保存');
    if(accountLabel)expect(screen.getByLabelText(accountLabel)).toHaveValue('');
    expect(document.body.textContent).not.toContain('fake-secret');
    expect(changed).toHaveBeenCalledOnce();
    fireEvent.click(within(form).getByRole('button',{name:'清除'}));
    await waitFor(()=>expect(clear).toHaveBeenCalledWith(provider));
    await waitFor(()=>expect(within(form).getByRole('button',{name:'清除'})).toBeDisabled());
    expect(within(form).getByRole('status')).toHaveTextContent('已清除');
    expect(changed).toHaveBeenCalledTimes(2);window.removeEventListener('credentials.changed',changed);
  });

  it('does not reflect server errors or claim saved state on a failed write',async()=>{
    server.use(http.put('/api/credentials/danbooru',()=>HttpResponse.json({error:{code:'older-server',message:'fake-secret and fake%2Dsecret and dGVzdF91c2VyOmZha2Utc2VjcmV0'}},{status:500})));
    const globalError=vi.fn();window.addEventListener('api.error',globalError);
    const changed=vi.fn();window.addEventListener('credentials.changed',changed);
    mount();await waitFor(()=>expect(screen.getAllByText('未配置')).toHaveLength(4));
    fireEvent.change(screen.getByLabelText('Danbooru 用户名'),{target:{value:'test_user'}});
    const secret=screen.getByLabelText('Danbooru API Key');fireEvent.change(secret,{target:{value:'fake-secret'}});
    fireEvent.click(within(secret.closest('form')!).getByRole('button',{name:'保存'}));
    expect(await within(secret.closest('form')!).findByRole('alert')).toHaveTextContent('服务器错误详情已隐藏');
    expect(screen.queryByRole('button',{name:'重试读取'})).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/fake-secret|fake%2Dsecret|dGVzdF91c2Vy/);
    expect(within(secret.closest('section')!).getByRole('status')).toHaveTextContent('未配置');
    expect(globalError).not.toHaveBeenCalled();expect(changed).not.toHaveBeenCalled();
    window.removeEventListener('api.error',globalError);window.removeEventListener('credentials.changed',changed);
  });

  it('does not convert a status failure to an anonymous or configured success',async()=>{
    let fail=true;
    server.use(http.get('/api/credentials',()=>fail?new HttpResponse(null,{status:503}):HttpResponse.json(configured)));
    mount();expect(await screen.findByRole('alert')).toHaveTextContent('无法读取密钥配置状态');
    expect(screen.getByRole('alert').closest('section')).toBeNull();
    expect(screen.getByRole('alert').compareDocumentPosition(document.getElementById('credentials-huggingface')!)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(screen.getAllByText('状态不可用')).toHaveLength(4);
    for(const button of screen.getAllByRole('button',{name:'清除'}))expect(button).toBeDisabled();
    fail=false;fireEvent.click(screen.getByRole('button',{name:'重试读取'}));
    await waitFor(()=>expect(screen.getAllByText('未配置')).toHaveLength(4));
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('keeps feedback with its provider when another provider fails to clear',async()=>{
    configured.gelbooru={configured:true};
    server.use(http.delete('/api/credentials/gelbooru',()=>HttpResponse.json({error:{message:'fake-secret'}},{status:503})));
    mount();await waitFor(()=>expect(screen.getAllByText('未配置')).toHaveLength(3));
    const hf=screen.getByLabelText('Hugging Face 访问令牌');
    const hfForm=within(hf.closest('form')!);
    fireEvent.change(hf,{target:{value:'fake-secret'}});
    fireEvent.click(hfForm.getByRole('button',{name:'保存'}));
    expect(await hfForm.findByRole('status')).toHaveTextContent('已保存');
    const gelForm=within(screen.getByLabelText('Gelbooru API Key').closest('form')!);
    fireEvent.click(gelForm.getByRole('button',{name:'清除'}));
    expect(await gelForm.findByRole('alert')).toHaveTextContent('凭据文件暂时不可读写');
    expect(gelForm.getByRole('button',{name:'清除'})).toBeEnabled();
    expect(gelForm.queryByRole('status')).not.toBeInTheDocument();
    expect(hfForm.getByRole('status')).toHaveTextContent('已保存');
    expect(hfForm.queryByRole('alert')).not.toBeInTheDocument();
    expect(document.body.textContent).not.toContain('fake-secret');
  });
});
