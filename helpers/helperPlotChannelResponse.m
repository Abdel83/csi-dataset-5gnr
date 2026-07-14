function helperPlotChannelResponse(Hest)
% helperPlotChannelResponse Plot channel response

figure
tiledlayout(2,2)
nexttile
waterfall(abs(Hest(:,:,1,1))')
xlabel("Subcarriers");
ylabel("Symbols");
zlabel("Channel Magnitude")
view(15,30)
colormap("cool")
title("(a): Rx=1, Tx=1")
nexttile
plot(squeeze(abs(Hest(:,1,:,1))))
grid on
xlabel("Subcarriers");
ylabel("Channel Magnitude")
%legend("Rx 1", "Rx 2")
title("(b): Symbol=1, Tx=1")
nexttile
waterfall(squeeze(abs(Hest(:,1,1,:)))')
view(-45,75)
grid on
xlabel("Subcarriers");
ylabel("Tx");
zlabel("Channel Magnitude")
title("(c): Symbol=1, Rx=1")
nexttile
nSubCarriers = size(Hest,1);
subCarrier = randi(nSubCarriers);
plot(squeeze(abs(Hest(subCarrier,1,:,:)))') 
grid on
xlabel("Tx");
ylabel("Channel Magnitude")
%legend("Rx 1", "Rx 2")
title("(d): Subcarrier=" + subCarrier + ", Symbol=1")
end