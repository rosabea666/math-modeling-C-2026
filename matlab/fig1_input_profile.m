clear; clc;

sourceFile = 'C:/Users/fzz17/OneDrive/Desktop/untitled.fig';
outDir = fileparts(mfilename('fullpath'));
src = openfig(sourceFile, 'invisible');
oldAxes = findall(src, 'Type', 'axes');
assert(numel(oldAxes) == 2, '此脚本适用于包含两个坐标轴的图1');

y = zeros(2, 1);
for k = 1:2
    oldAxes(k).Units = 'normalized';
    y(k) = oldAxes(k).Position(2);
end
[~, order] = sort(y, 'descend');
oldAxes = oldAxes(order);

fig = figure('Units', 'centimeters', ...
    'Position', [3 3 16 11.5], 'Color', 'w', 'Renderer', 'painters');
ax1 = copyobj(oldAxes(1), fig);
ax2 = copyobj(oldAxes(2), fig);
close(src);

fontCN = 'SimSun';
fontEN = 'Times New Roman';
ink = [0.22 0.24 0.27];

for ax = [ax1 ax2]
    ax.Units = 'normalized';
    ax.PositionConstraint = 'innerposition';
    ax.Title.String = '';
    ax.FontName = fontEN;
    ax.FontSize = 9;
    ax.LineWidth = 0.7;
    ax.Box = 'off';
    ax.TickDir = 'out';
    ax.TickLength = [0.008 0.008];
    ax.XColor = ink;
    ax.YColor = ink;
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    ax.GridAlpha = 0.15;
    ax.Layer = 'top';
    ax.YAxis.Exponent = 0;
    ax.XLim = [0 24];
    ax.XTick = 0:4:24;
    ax.XTickLabel = {'00:00','04:00','08:00','12:00','16:00','20:00','24:00'};
    ax.YLabel.FontName = fontCN;
    ax.YLabel.FontSize = 10;
end

ax1.Position = [0.14 0.53 0.81 0.32];
ax2.Position = [0.14 0.12 0.81 0.25];
ax1.XTickLabel = [];
ax1.XLabel.String = '';

ylabel(ax1, '功率 (kW)', 'FontName', fontCN, 'FontSize', 10);
ylabel(ax2, '电价 (元/kWh)', 'FontName', fontCN, 'FontSize', 10);
xlabel(ax2, '时刻', 'FontName', fontCN, 'FontSize', 10);

annotation(fig, 'textbox', [0.14 0.935 0.81 0.035], ...
    'String', '(a) 日内负载与光伏供需关系', ...
    'FontName', fontCN, 'FontSize', 10, 'Color', ink, ...
    'EdgeColor', 'none', 'Margin', 0, 'VerticalAlignment', 'middle');

annotation(fig, 'textbox', [0.14 0.415 0.81 0.035], ...
    'String', '(b) 日内购电价格', ...
    'FontName', fontCN, 'FontSize', 10, 'Color', ink, ...
    'EdgeColor', 'none', 'Margin', 0, 'VerticalAlignment', 'middle');

hold(ax1, 'on');
h1 = plot(ax1, NaN, NaN, '-', 'Color', [0 0.447 0.698], 'LineWidth', 1.25);
h2 = plot(ax1, NaN, NaN, '--', 'Color', [0.835 0.369 0], 'LineWidth', 1.25);
h3 = patch(ax1, NaN, NaN, [0.865 0.930 0.885], 'EdgeColor', 'none');
lgd = legend(ax1, [h1 h2 h3], ...
    {'小区负载', '光伏预测', '潜在光伏富余'}, ...
    'Orientation', 'horizontal', 'NumColumns', 3, ...
    'Box', 'off', 'FontName', fontCN, 'FontSize', 9);
lgd.AutoUpdate = 'off';
lgd.ItemTokenSize = [22 10];
lgd.Units = 'normalized';
lgd.Position = [0.19 0.872 0.71 0.040];

ax1.Position = [0.14 0.53 0.81 0.32];
ax2.Position = [0.14 0.12 0.81 0.25];
linkaxes([ax1 ax2], 'x');
drawnow;

savefig(fig, fullfile(outDir, 'fig1_layout_fixed.fig'));
exportgraphics(fig, fullfile(outDir, 'fig1_layout_fixed.pdf'), ...
    'ContentType', 'vector', 'BackgroundColor', 'white');
exportgraphics(fig, fullfile(outDir, 'fig1_layout_fixed.png'), ...
    'Resolution', 600, 'BackgroundColor', 'white');