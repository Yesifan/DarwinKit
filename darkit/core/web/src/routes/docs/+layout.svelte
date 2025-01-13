<script>
	import Navigation from './navigation.json';
	import ZhNavigation from './navigation_zh.json';
	import { languageTag } from '$lib/paraglide/runtime';
	import DocsNavbar from '$lib/components/docs-navbar.svelte';

	let { children } = $props();

	const navigation = $derived(languageTag() === 'zh' ? ZhNavigation : Navigation);

	const docsContents = $derived(
		navigation.map((item) => {
			return {
				caption: item.title,
				contents: item.children.map((child) => {
					const title = child.metadata.title || child.title;
					return [title, child.path];
				})
			};
		})
	);
</script>

<DocsNavbar contents={docsContents} />

<div class="w-full overflow-y-auto px-12 py-8">
	<article class="prose mx-auto max-w-6xl">
		{@render children()}
	</article>
</div>
