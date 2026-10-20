# 枕头形变参考（2026-09-08）

这些视频只用于理解形状和动作，不复制视频到交付物，不把未知手力、未知播放倍率推算成精确材料常数。

- Down & Feather Company：How to fluff a flat feather pillow，https://www.youtube.com/watch?v=ECGcCL7T0pw 。浏览器实际查看 8:34、8:39 的双手挤压演示关键帧；抓持端出现集中褶皱，枕芯仍保持蓬松，边缘收薄，外形有明显不规则鼓胀。原作者文字说明手风琴式揉松用于重新分散填充。参考的是松散填充枕的外形与整理方式，不能由这些画面测得准确回弹时间。配套文字 https://www.downandfeathercompany.com/blogs/news/how-to-fluff-a-flat-feather-pillow （网页浏览器被拦，但搜索阅读服务可读取文字，视频通过 YouTube 公开页面查看）。
- JKshabamzayy：Super EPIC Slow Motion Pillow Fight，https://www.youtube.com/watch?v=_iJePboLd40 。实际查看约 0:59.8、1:04.8 的关键帧：手抓住局部，枕体大幅弯曲，受力端变薄，外套有拉伸方向的褶皱。慢动作倍率和填充规格未知，只作为弯折和高速接触场景参考。不是材料测量实验。
- Sleepopolis：Brooklyn Bedding Talalay Latex Pillow Review，https://www.youtube.com/watch?v=1Vy9fgT8RGY 。实际通过播放器逐帧查看 3:16.22、3:16.72、3:17.47：双手施压时产生两个局部凹陷，手离开后恢复平滑鼓起的外形，周边仍维持较饱满的轮廓；据此把 foam 作为较有支撑、较快回弹的对照。仅有画面和播放时间，不能据此声称精确杨氏模量或完整回弹曲线已经标定。

## 当前物理表达范围

同一 Isaac Sim 5.1 / IsaacLab 2.3.2 的可变形体求解器，原创新模型；soft 为易弯折填充枕的等效连续体，foam 为弹性泡棉枕的等效连续体。均不是羽毛颗粒迁移模型，也不是带温度依赖、塑性记忆的真实 TEMPUR 材料。参数需用压缩/释放测试约束，并明确尚未做力传感实测标定。
