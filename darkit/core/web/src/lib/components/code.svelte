<script lang="ts">
	import { cn } from '$lib/utils';
	import Prism from 'prismjs';
	import 'prismjs/components/prism-bash';
	import 'prismjs/components/prism-python';
	import CopyClipboard from './copy-clipboard.svelte';
	import ScrollArea from './ui/scroll-area/scroll-area.svelte';

	type Props = {
		lang: 'py' | 'js' | 'bash';
		wrap?: boolean;
		content: string;
		class?: string;
	};
	let { lang, content, wrap, class: className }: Props = $props();
	let codeHtml = $derived(Prism.highlight(content, Prism.languages[lang], lang));
	// Add your component logic here
</script>

<ScrollArea class={cn('scroll-y-auto relative h-full', className)} orientation="vertical">
	<pre class={cn('h-full w-full', `language-${lang}`)}><code
			class={cn(`language-${lang}`, { '!whitespace-pre-wrap': wrap })}>{@html codeHtml}</code
		>
		<CopyClipboard text={content} class="absolute bottom-2 right-2" />
	</pre>
</ScrollArea>
