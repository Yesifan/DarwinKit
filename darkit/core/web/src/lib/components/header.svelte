<script lang="ts">
	import { page } from '$app/stores';
	import { cn } from '$lib/utils';
	import logo from '$lib/images/logo.svg';
	import DarkModeButton from './dark-mode-button.svelte';
	import I18nButton from './i18n-button.svelte';

	let pathname = $derived($page.url.pathname);
</script>

{#snippet item(name: string, path: string, root: string | null = null)}
	<a
		href={path}
		aria-current={pathname === path}
		class={cn(
			'hover:text-theme px-1',
			pathname.includes(root ?? path) && 'border-primary border-b-2'
		)}
	>
		{name}
	</a>
{/snippet}

<header
	class="border-border/40 bg-background/90 supports-[backdrop-filter]:bg-background/60 flex h-14 w-full flex-shrink-0 items-center border-b px-8 dark:border-gray-800"
>
	<a class="mr-8 flex items-center" href="/">
		<img src={logo} alt="logo" class="h-8" />
		<span class="text-primary px-2 text-lg font-bold"> DarwinKit </span>
	</a>
	<ul class="flex flex-1 gap-2">
		{@render item('Doc', '/docs', '/docs')}
		{@render item('LM', '/lm/train', '/lm')}
		{@render item('Build Network', '/flow', '/flow')}
	</ul>
	<I18nButton class="mr-4" />
	<DarkModeButton />
</header>
