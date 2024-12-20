<script>
	import DocsNavbar from '$lib/components/docs-navbar.svelte';
	import Navigation from './navigation.json';

	let { children } = $props();

	const docsContents = Navigation.map((item) => {
		return {
			caption: item.title,
			contents: item.children.map((child) => {
				return [child.title, child.path];
			})
		};
	});
</script>

<DocsNavbar contents={docsContents} />

<div class="w-full overflow-y-auto px-12 py-8">
	<article class="prose mx-auto max-w-6xl">
		{@render children()}
	</article>
</div>
