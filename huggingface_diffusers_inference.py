# 亢体造梦 2023 copyright
from diffusers import DDPMPipeline
from PIL import Image
import minepi
import os
import asyncio

def count_files(directory):
    '''
    计算目录下文件数量
    :param directory: 目录
    :return: 文件数量
    '''
    return sum(len(files) for _, _, files in os.walk(directory))

async def render_minecraft_skin(minecraft_skin):
    await minecraft_skin.render_skin()
    return minecraft_skin.skin

if __name__ == '__main__':
    # 确保skin文件夹存在
    skin_folder = './skin'
    os.makedirs(skin_folder, exist_ok=True)

    # 使用huggingface的diffusers库渲染皮肤
    pipeline = DDPMPipeline.from_pretrained('./mcskin_diffuser_0609').to('cpu')
    image = pipeline().images[0].convert('RGBA')

    # 逐个像素判断是否为黑色，若为黑色则转换成透明像素
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            if image.getpixel((x, y))[3] < 115:
                image.putpixel((x, y), (255, 255, 255, 0))

    # 保存皮肤到skin文件夹
    files_num = count_files(skin_folder) + 1
    image.save(os.path.join(skin_folder, f'skin{files_num}.png'))

    # 使用minepi库渲染并生成皮肤
    skin_object = minepi.Skin(raw_skin=image, raw_cape=None, raw_skin_url=None, raw_cape_url=None, name=None)
    asyncio.run(render_minecraft_skin(skin_object))
    skin_object.skin.show()
