<script lang="ts">
	export let command = 'apt-get --force true update';

	// 将命令分割成数组并添加类型
	function parseCommand(command: string) {
		const parts = command.split(' ');
		return parts.map((part, index) => {
			if (index === 0) {
				return { text: part, type: 'main' };
			} else if (part.startsWith('--') || part.startsWith('-')) {
				return { text: part, type: 'secondary' };
			} else if (parts[index - 1].startsWith('--') || parts[index - 1].startsWith('-')) {
				return { text: part, type: 'tertiary' };
			} else {
				return { text: part, type: 'default' };
			}
		});
	}

	$: parsedCommand = parseCommand(command);
</script>

<div
	class="bg-gray min-h-48 space-x-3 rounded-md bg-gray-950 p-4 font-mono text-white dark:bg-gray-800"
>
	{#each parsedCommand as part}
		<span
			class={part.type === 'main'
				? 'text-emerald-400'
				: part.type === 'secondary'
					? 'text-white'
					: part.type === 'tertiary'
						? 'text-white/60'
						: 'text-sky-500'}
		>
			{part.text}
		</span>
	{/each}
</div>
