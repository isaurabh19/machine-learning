def bootstrapping(model, X,Y,B)
	n_iterations = 100
	n_size = 100
	values = np.column_stack((X,Y))
	roc_scores = []
	accuracy_scores = []
	for i in range(n_iterations):
		train = resample(values, n_samples=n_size)
		test = np.array([x for x in values if x.tolist() not in train.tolist()])
		model.fit(train[:,:-1], train[:,-1])
		y_scores = model.predict(test[:,:-1])
		roc_scores.append(roc_auc_score(test[:,-1], y_scores))
		accuracy_scores.append(accuracy_score(test[:,-1], y_scores))
	alpha = 0.68
	p = ((1.0-alpha)/2.0) * 100
	lower = max(0.0, np.percentile(roc_scores, p))
	p = (alpha+((1.0-alpha)/2.0)) * 100
	upper = min(1.0, np.percentile(roc_scores, p))
	print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
return np.mean(roc_scores), np.std(roc_scores), np.mean(accuracy_scores), np.std(accuracy_scores)

