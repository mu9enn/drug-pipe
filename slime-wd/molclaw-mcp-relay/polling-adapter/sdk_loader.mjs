import { pathToFileURL } from 'node:url';
const root=process.env.DRUG_PROJECT;
if(!root)throw new Error('DRUG_PROJECT is required');
export const sdk=process.env.MOLCLAW_DEPS_ROOT ? process.env.MOLCLAW_DEPS_ROOT+'/node_modules/@modelcontextprotocol/sdk/dist/esm/' : root+'/slime-wd/deepseek-harness/node_modules/.pnpm/@modelcontextprotocol+sdk@1.29.0_zod@4.4.3/node_modules/@modelcontextprotocol/sdk/dist/esm/';
export const load=path=>import(pathToFileURL(sdk+path));
