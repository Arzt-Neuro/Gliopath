import openslide
import matplotlib.pyplot as plt

def vue_slide(path):
    # Open the slide
    slide = openslide.OpenSlide(path)

    # Get a thumbnail for overview (fast)
    thumbnail = slide.get_thumbnail((1024, 1024))

    # Display thumbnail
    plt.figure(figsize=(10, 10))
    plt.imshow(thumbnail)
    plt.axis('off')
    plt.title('NDPI Slide Overview')
    plt.show()



def vue_region(path, loc=(5000,5000), res=0, scale=(1000,1000)):
    # Open the slide
    slide = openslide.OpenSlide(path)

    # For detailed regions, read specific areas
    # (x, y, width, height) at level 0 (highest resolution)
    region = slide.read_region(loc, res, scale)
    region_rgb = region.convert('RGB')

    plt.figure(figsize=(8, 8))
    plt.imshow(region_rgb)
    plt.axis('off')
    plt.title('Detailed Region')
    plt.show()

def montrer_meta(path):
    # Open the slide
    slide = openslide.OpenSlide(path)

    """全面分析OpenSlide对象"""

    print("=" * 60)
    print("OPENSLIDE 对象全面分析")
    print("=" * 60)

    # 1. 对象基本信息
    print("1. 基本信息")
    print("-" * 20)
    print(f"对象类型: {type(slide)}")
    print(f"对象ID: {id(slide)}")

    # 2. 几何结构
    print("\n2. 几何结构")
    print("-" * 20)
    width, height = slide.dimensions
    print(f"总尺寸: {width:,} × {height:,} 像素")
    print(f"总像素: {width * height:,}")
    print(f"宽高比: {width/height:.2f}:1")
    print(f"层级数: {slide.level_count}")

    # 详细层级信息
    print(f"\n层级详情:")
    for i in range(slide.level_count):
        w, h = slide.level_dimensions[i]
        downsample = slide.level_downsamples[i]
        pixels = w * h
        mb_estimate = pixels * 3 / (1024**2)  # RGB估算
        print(f"  Level {i}: {w:6,} × {h:6,} | {downsample:6.1f}x | ~{mb_estimate:5.1f}MB")

    # 3. 扫描参数
    print("\n3. 扫描参数")
    print("-" * 20)
    key_properties = {
        "扫描仪": "openslide.vendor",
        "格式": "openslide.format",
        "物镜倍数": "openslide.objective-power",
        "X轴分辨率": "openslide.mpp-x",
        "Y轴分辨率": "openslide.mpp-y",
        "背景色": "openslide.background-color",
        "边界": "openslide.bounds-x",
    }

    for label, key in key_properties.items():
        value = slide.properties.get(key, "未找到")
        print(f"  {label}: {value}")

    # 计算物理尺寸
    mpp_x = slide.properties.get("openslide.mpp-x")
    mpp_y = slide.properties.get("openslide.mpp-y")
    if mpp_x and mpp_y:
        try:
            mpp_x, mpp_y = float(mpp_x), float(mpp_y)
            phys_w = width * mpp_x / 1000  # 转换为mm
            phys_h = height * mpp_y / 1000
            print(f"  物理尺寸: {phys_w:.1f} × {phys_h:.1f} mm")
        except:
            pass

    # 4. 厂商特定属性
    print("\n4. 厂商特定属性")
    print("-" * 20)
    vendor_keys = [k for k in slide.properties.keys() if not k.startswith("openslide.")]
    vendor_groups = {}
    for key in vendor_keys:
        prefix = key.split('.')[0]
        if prefix not in vendor_groups:
            vendor_groups[prefix] = []
        vendor_groups[prefix].append(key)

    for prefix, keys in vendor_groups.items():
        print(f"  {prefix.upper()} ({len(keys)} 个属性):")
        for key in keys[:3]:  # 只显示前3个
            value = slide.properties[key]
            if len(str(value)) > 50:
                value = str(value)[:50] + "..."
            print(f"    {key}: {value}")
        if len(keys) > 3:
            print(f"    ... 还有 {len(keys)-3} 个属性")

    # 5. 方法列表
    print("\n5. 可用方法")
    print("-" * 20)
    methods = [method for method in dir(slide) if not method.startswith('_') and callable(getattr(slide, method))]
    for method in methods:
        print(f"  {method}()")

    # 6. 内存和性能信息
    print("\n6. 性能信息")
    print("-" * 20)
    total_pixels_level0 = width * height
    estimated_ram_gb = total_pixels_level0 * 3 / (1024**3)
    print(f"  Level 0 完整图像估算: {estimated_ram_gb:.2f} GB")

    # 推荐的使用层级
    for task, level_range in [("快速预览", "7-8"), ("组织分析", "3-5"), ("细胞分析", "0-2")]:
        print(f"  {task}: 推荐使用 Level {level_range}")

    return slide
