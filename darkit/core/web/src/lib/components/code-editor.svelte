<script lang="ts" module>
	import { cn } from '$lib/utils';
	import Prism from 'prismjs';
	import 'prismjs/components/prism-bash';
	import 'prismjs/components/prism-python';
	import type { CodeJar } from 'codejar';
	import type { Action } from 'svelte/action';

	type Lang = 'py' | 'js' | 'bash';
	type CodeJarOpt = Parameters<typeof CodeJar>[2];
	type onChange = (e: CustomEvent<string>) => void;
	type baseProps = {
		code: string;
		lang?: Lang;
		class?: string;
		onChange?: onChange;
	};
	type CoreJarProps = baseProps & {
		options: CodeJarOpt;
	};

	export const corejar: Action<HTMLElement, CoreJarProps> = (
		node,
		{ code, lang = 'py', onChange, options }
	) => {
		let editor: CodeJar | null = null;

		import('codejar').then(({ CodeJar }) => {
			const hl = function (el: HTMLElement) {
				el.innerHTML = Prism.highlight(el.textContent ?? '', Prism.languages[lang], lang);
			};

			editor = CodeJar(node, hl, options);
			editor.updateCode(code);

			editor.onUpdate((code) => {
				const e = new CustomEvent('change', { detail: code });
				onChange?.(e);
			});
		});

		function update({ code }: CoreJarProps) {
			if (editor && code !== editor.toString()) {
				editor.updateCode(code);
			}
		}

		return {
			update,
			destroy() {
				if (editor) {
					editor.destroy();
				}
			}
		};
	};
</script>

<script>
	type Props = { onChange: onChange } & baseProps & CodeJarOpt;

	let { code, lang = 'py', class: className, onChange, ...options }: Props = $props();
</script>

<pre class={cn('relative', `language-${lang}`, className)}><code
		class={cn(`language-${lang}`)}
		use:corejar={{ code, lang, onChange, options }}></code>
</pre>
