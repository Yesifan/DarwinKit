import type { AvailableLanguageTag } from '$lib/paraglide/runtime';
import type { ParaglideLocals } from '@inlang/paraglide-sveltekit';
// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
declare global {
	declare module '*.md' {
		const content: jsx.JSX.Element;
		export default content;
	}
	namespace App {
		// interface Error {}
		interface Locals {
			paraglide: ParaglideLocals<AvailableLanguageTag>;
		}
		// interface PageData {}
		// interface PageState {}
		// interface Platform {}
	}
}

export {};
