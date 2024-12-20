// src/lib/i18n.ts
import { goto } from '$app/navigation';
import { createI18n } from '@inlang/paraglide-sveltekit';
import * as runtime from '$lib/paraglide/runtime.js';

export const i18n = createI18n(runtime, {
	exclude: [/^\/docs\/\d+$/]
});

export const gotoWithi18n = (url: string) => {
	goto(i18n.resolveRoute(url));
};
