import {render, screen, within, fireEvent, waitFor} from '@testing-library/react';
import {afterEach, expect, it, vi} from 'vitest';
import {SchemaForm} from '../../../frontend/src/schema/SchemaForm/SchemaForm';
import schema from '../../../frontend/src/schema/train-schema.json';
import {apiClient} from '../../../frontend/src/api/client';
import i18n from '../../../frontend/src/i18n';

afterEach(() => vi.restoreAllMocks());
it('offers a detected shared VAE and browses the configured component category', async () => {
  await i18n.changeLanguage('zh-CN');
  const get = vi.spyOn(apiClient,'get').mockImplementation(async (endpoint, options) => {
    if(endpoint==='/models') return [{id:'vae',family:'krea2',kind:'vae',path:'/models/vae/shared/qwen.safetensors',exists:true,compatible_families:['anima','krea2']}] as any;
    if(endpoint==='/models/browse-root') return {path:`/custom/models/${options?.params?.kind==='vae'?'vae':'diffusion_models'}`} as any;
    if(endpoint==='/fs/list') return {path:'/custom/models/vae',parent:'/custom/models',entries:[]} as any;
    return [] as any;
  });
  render(<SchemaForm schema={schema} value={{model:{family:'anima'}}} onChange={()=>{}} groupFilter={['model']} compact/>);
  const field=screen.getByTestId('field-model.vae_path');
  const select=await within(field).findByTestId('model-registry-select');
  fireEvent.click(select);
  expect(await screen.findByRole('option',{name:'qwen.safetensors'})).toBeVisible();
  fireEvent.keyDown(select,{key:'Escape'});
  fireEvent.click(within(field).getByRole('button',{name:'浏览'}));
  await waitFor(()=>expect(get).toHaveBeenCalledWith('/fs/list',expect.objectContaining({params:{path:'/custom/models/vae'}})));
});
