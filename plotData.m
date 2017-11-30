function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

q = [X y];
X1 = [0 0];
X0 = [0 0];

% Separates the students grades by
for i = 1:size(q,1)
	if q(i,3) == 1
		X1 = [X1; q(i,1:2)];
	else
		X0 = [X0; q(i,1:2)];
	end
end

% removes the first row (added when created)
X1 = X1(2:size(X1,1), :);
X0 = X0(2:size(X0,1), :);

% plots the grades with different markers
plot(X1(:,1), X1(:,2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X0(:,1), X0(:,2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);


hold off;

end
