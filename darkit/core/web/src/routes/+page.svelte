<script lang="ts">
	import { languageTag } from '$lib/paraglide/runtime';
	import Home from './home.md';
	import HomeZH from './home.zh.md';

	let lang = languageTag();
</script>

<div class="w-full overflow-y-auto p-16">
	<article class="prose mx-auto">
		<h1 class="text-theme w-full py-4 text-center text-8xl font-bold">DarwinKit</h1>
		{#if lang === 'zh'}
			<HomeZH />
		{:else}
			<Home />
		{/if}
	</article>
</div>
