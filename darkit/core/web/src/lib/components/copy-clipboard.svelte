<script lang="ts">
	import { Copy } from 'lucide-svelte';
	import { Button, type Props } from '$lib/components/ui/button/index.js';
	import { copyText } from '$lib/utils';
	import { toast } from 'svelte-sonner';

	type $$Props = Props & { text?: string };

	// 函数：将文本复制到剪贴板
	function copyToClipboard() {
		copyText(text)
			.then(() => {
				toast.success('Copied to clipboard');
			})
			.catch((err) => {
				toast.error('Copy failed, please manually copy the text');
			});
	}

	export let text: string = '';
</script>

<Button variant="outline" size="icon" onclick={copyToClipboard} {...$$restProps}>
	<Copy />
</Button>
