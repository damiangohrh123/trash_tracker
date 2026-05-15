import 'package:flutter_test/flutter_test.dart';

import 'package:trash_tracker/main.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('App shows title while loading', (WidgetTester tester) async {
    debugSetCamerasForTest([]);
    await tester.pumpWidget(const TrashTrackerApp());
    expect(find.text('Trash Tracker'), findsOneWidget);
  });
}
