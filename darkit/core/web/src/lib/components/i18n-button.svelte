<script lang="ts">
	import Languages from 'lucide-svelte/icons/languages';
	import * as Popover from '$lib/components/ui/popover';
	import Button from './ui/button/button.svelte';
	import { availableLanguageTags, languageTag } from '$lib/paraglide/runtime.js';
	import { i18n } from '$lib/i18n';
	import { page } from '$app/stores';

	let className: string | undefined = undefined;
	export { className as class };
</script>

<Popover.Root>
	<Popover.Trigger class={className}>
		<Languages class="h-[1.2rem] w-[1.2rem]" />
	</Popover.Trigger>
	<Popover.Content class="flex w-28 flex-col p-2">
		{#each availableLanguageTags as lang}
			<!-- the hreflang attribute decides which language the link points to -->
			<Button variant="ghost" class="justify-start px-2 py-1" disabled={lang === languageTag()}>
				<a
					href={i18n.route($page.url.pathname)}
					hreflang={lang}
					aria-current={lang === languageTag() ? 'page' : undefined}
				>
					{#if lang === 'en'}
						English
					{:else if lang === 'zh'}
						简体中文
					{:else}
						{lang}
					{/if}
				</a>
			</Button>
		{/each}
	</Popover.Content>
</Popover.Root>
