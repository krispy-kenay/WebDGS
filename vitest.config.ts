import { defineConfig, mergeConfig } from 'vitest/config'
import { webdriverio } from '@vitest/browser-webdriverio'
import viteConfig from './vite.config'

export default mergeConfig(
    viteConfig,
    defineConfig({
        test: {
            browser: {
                enabled: true,
                provider: webdriverio({
                    capabilities: {
                        browserName: 'chrome',
                        'goog:chromeOptions': {
                            args: [
                                //'--headless=new',
                                '--enable-unsafe-webgpu',
                                '--enable-features=Vulkan',
                                '--use-angle=vulkan',
                                '--disable-gpu-sandbox',
                            ],
                        },
                    },
                }),
                instances: [{ browser: 'chrome' }],
                headless: false,
            },
        },
        resolve: {
            alias: {
                '@/': new URL('./src/', import.meta.url).pathname,
            },
        },
    })
)
