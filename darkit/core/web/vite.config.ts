import { defineConfig } from 'vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { paraglide } from '@inlang/paraglide-sveltekit/vite';
import { docs } from './vite-plugin-docs';

export default defineConfig({
	plugins: [
		docs({
			source: '../../../docs',
			target: './src/routes/docs',
			static: './static/docs'
		}),
		paraglide({
			//recommended
			project: './project.inlang',
			outdir: './src/lib/paraglide'
		}),
		sveltekit()
	],
	server: {
		proxy: {
			'/api': { target: 'http://localhost:8000', ws: true }
		}
	}
});
