import path from 'path';
import fs from 'fs/promises';
import fm from 'front-matter';
import { watch, type FSWatcher } from 'fs';
import type { Plugin } from 'vite';

export const DOCS_NAVIGATION = 'navigation.json';
export const ZH_DOCS_NAVIGATION = 'navigation_zh.json';

const ROOT_PATH = path.join(__dirname, '../../../');
const ROOT_STATIC_PATH = path.join(ROOT_PATH, 'static/docs');

type DocsTree =
	| {
			order: number;
			title: string;
			children: DocsTree[];
	  }
	| {
			order: number;
			title: string;
			path: string;
			metadata: Record<string, any>;
	  };

const isPathIn = (a: string, b: string) => {
	const relative = path.relative(b, a);
	return !relative.startsWith('..') && !path.isAbsolute(relative);
};

const createNavigation = async (
	source: string,
	root: string | undefined = undefined
): Promise<DocsTree[]> => {
	const docsTree: DocsTree[] = [];

	const files = await fs.readdir(source, { withFileTypes: true });
	for (const file of files) {
		//Generate a navigation from an en document
		const [order, title] = file.name.split('.');
		// 跳过不符合“序号.标题”格式的文件或目录
		if (!order || !title) continue;

		if (file.isDirectory()) {
			const root2 = root ? `${root}/${title}` : title;
			const childrenTree = await createNavigation(path.join(source, file.name), root2);
			docsTree.push({
				order: parseInt(order, 10),
				title: title.replace(/-/g, ' '),
				children: childrenTree
			});
		} else if (file.isFile() && file.name.endsWith('.md')) {
			const content = await fs.readFile(path.join(source, file.name), 'utf-8');
			const { attributes } = fm(content);

			const title2 = title.replace('.md', '');
			// svelte router path
			const urlPath = root ? `/docs/${root}/${title2}` : `/docs/${title2}`;
			docsTree.push({
				order: parseInt(order, 10),
				title: title2.replace(/-/g, ' '),
				path: urlPath,
				metadata: attributes as Record<string, any>
			});
		}
	}

	return docsTree;
};

const removeDocsCache = async (target: string) => {
	const files = await fs.readdir(target, { withFileTypes: true });

	for (const file of files) {
		const filePath = path.join(target, file.name);
		if (file.isDirectory()) {
			await removeDocsCache(filePath);
			const remainingFiles = await fs.readdir(filePath);
			if (remainingFiles.length === 0) {
				await fs.rmdir(filePath);
			}
		} else if (file.isFile() && file.name.endsWith('.md')) {
			await fs.unlink(filePath);
		}
	}
};

/**
 * 将 markdown 文件中的静态资源引用转换为 web 路径，将链接路径页转换为 web 路径
 * @returns { content: string, staticFiles: [string, string][] } 转换后的 markdown 内容和静态资源文件路径
 * staticFiles 为 [源文件路径, 相对路径] 的数组
 */
const transformMarkdown = async (sourceDir: string) => {
	const content = await fs.readFile(sourceDir, 'utf-8');
	const staticFiles: [string, string][] = [];
	// 将 markdown 文件中的本地静态资源引用转换为 web 路径， 并将本地文件路径读取出来
	const updatedContent = content.replace(/!\[.*?\]\((\/.*?)(?:\s".*?")?\)/g, (match, p1) => {
		const staticFile = path.isAbsolute(p1) ? path.join(ROOT_PATH, p1) : path.join(sourceDir, p1);
		// 如果是网络文件则跳过
		if (p1.startsWith('http')) return match;
		if (isPathIn(staticFile, ROOT_STATIC_PATH)) {
			const relativePath = path.relative(ROOT_STATIC_PATH, staticFile);
			staticFiles.push([staticFile, relativePath]);
			return match.replace(p1, `/docs/${relativePath}`);
		} else {
			throw new Error(
				`Docs static file ${staticFile} must be in ${ROOT_STATIC_PATH}, but it is not.`
			);
		}
	});
	// 将 markdown 文件中的链接路径转换为 web 路径
	const finalContent = updatedContent.replace(/\[.*?\]\((.*?).md\)/g, (match, p1) => {
		// 如果是网络文件则跳过
		if (p1.startsWith('http')) return match;
		const newUrl = (p1 as string)
			.split('/')
			.map((segment) => {
				if (segment !== '.' && segment !== '..' && segment.includes('.')) {
					return segment.split('.')[1];
				}
				return segment;
			})
			.join('/');

		return match.replace(`${p1}.md`, newUrl);
	});
	return { content: finalContent, staticFiles };
};

/**
 * 文件夹和文档的命名需要符合“序号.标题”格式
 * 构建到 svelte 中时根据序号排序，标题作为 svelte 文件夹和组件的名称
 */
export const docs = ({
	source,
	target,
	static: staticDir
}: {
	source: string;
	target: string;
	static: string;
}): Plugin => {
	let isDev = false;
	let watcher: FSWatcher | null = null;
	const sourcePath = path.join(__dirname, source);
	const targetPath = path.join(__dirname, target);
	const docStaticPath = path.join(__dirname, staticDir);
	const navJSONPath = path.join(targetPath, DOCS_NAVIGATION);
	const zhNavJSONPath = path.join(targetPath, ZH_DOCS_NAVIGATION);

	const i18nWriteNavigationFile = async () => {
		const enDocsTree = await createNavigation(path.join(ROOT_PATH, 'docs'));
		const zhDocsTree = await createNavigation(path.join(ROOT_PATH, 'docs', 'zh'));
		await fs.writeFile(navJSONPath, JSON.stringify(enDocsTree));
		await fs.writeFile(zhNavJSONPath, JSON.stringify(zhDocsTree));
	};

	/** 复制静态文件 */
	const copyStaticFiles = async (files: [string, string][]) => {
		for (const [sourceFile, relativePath] of files) {
			const targetFile = path.join(docStaticPath, relativePath);
			await fs.mkdir(path.dirname(targetFile), { recursive: true });
			await fs.copyFile(sourceFile, targetFile);
		}
	};

	/** 递归复制 docs 目录下的文件到 target 目录 */
	const copyDocsPage = async (source: string, target: string) => {
		const files = await fs.readdir(source, { withFileTypes: true });

		for (const file of files) {
			// lang dir skip
			const isLangDir = ['zh'].includes(file.name);
			const [order, title] = file.name.split('.');
			if ((!order || !title) && !isLangDir) continue; // 跳过不符合“序号.标题”格式的文件或目录

			if (file.isDirectory()) {
				// target 目录名仅保留标题部分,
				await copyDocsPage(path.join(source, file.name), path.join(target, title ?? file.name));
			} else if (file.isFile() && file.name.endsWith('.md')) {
				// 源文档文件写入到目标文件
				const targetDir = path.join(target, title);
				const targetFile = path.join(targetDir, '+page.md');
				await fs.mkdir(targetDir, { recursive: true });
				const { content, staticFiles } = await transformMarkdown(path.join(source, file.name));
				await copyStaticFiles(staticFiles);
				await fs.writeFile(targetFile, content);
			}
		}
	};

	/** 监听 docs 目录下的文件变化，实时更新 docs 目录 */
	const buildDocsPage = (source: string, target: string) => {
		const events = new Set<string>();
		let timer: NodeJS.Timeout | null = null;

		const watcher = watch(source, { recursive: true }, (_event, filename) => {
			if (filename) events.add(filename);
			if (timer) clearTimeout(timer);
			// 事件去重，合并0.5秒的所有文件修改事件
			timer = setTimeout(async () => {
				const cache = new Set<string>(events);
				events.clear();
				console.debug(`Docs page ${Array.from(cache).join(',')} changed: rebuild...`);
				// 写入新的文档目录
				await i18nWriteNavigationFile();
				cache.forEach(async (filename) => {
					const sourceFile = path.join(source, filename);
					if (filename.endsWith('.md')) {
						// 将文件名转换为 svelte router path
						const fpath = filename
							.replace('.md', '')
							.split(path.sep)
							.map((p) => (p.includes('.') ? p.split('.')[1] : p))
							.join('/');
						const sourceIsExist = await fs.access(sourceFile).catch(() => false);
						const targetFile = path.join(target, fpath, '+page.md');
						const targetIsExist = await fs.access(targetFile).catch(() => false);
						if (sourceIsExist !== false) {
							// 源文档文件写入到目标文件
							const targetDir = path.dirname(targetFile);
							await fs.mkdir(targetDir, { recursive: true });
							const { content, staticFiles } = await transformMarkdown(sourceFile);
							await copyStaticFiles(staticFiles);
							await fs.writeFile(targetFile, content);
						} else if (targetIsExist !== false) {
							// 如果源文件不存在，删除目标文件
							await fs.unlink(targetFile);
						}
					}
				});
			}, 500);
		});
		return watcher;
	};

	return {
		name: 'docs-plugin',
		async config(config, { command }) {
			isDev = command === 'serve';
			// 写入新的文档目录
			await i18nWriteNavigationFile();
			await removeDocsCache(targetPath);
			await copyDocsPage(sourcePath, targetPath);
		},
		async buildStart() {
			console.log('Copy the docs page...');
			console.log('Docs page copied.');

			if (isDev && watcher === null) {
				watcher = watch(source, { recursive: true });
				console.log('Start watching docs......');
				watcher = buildDocsPage(sourcePath, targetPath);
			}
		},
		async buildEnd() {
			if (watcher) {
				watcher.close();
			}
		}
	};
};
