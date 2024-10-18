import sys
import json
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QCheckBox, QPushButton, QFileDialog, 
                             QScrollArea, QComboBox, QSpinBox, QTextEdit, QInputDialog, QMessageBox,
                             QSplitter, QSizePolicy, QListWidget, QAbstractItemView, QListWidgetItem)

from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QTransform
import graphviz

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Genetic Algorithm Configuration")
        self.setGeometry(100, 100, 1920, 998)  # Increased initial size
        self.showMaximized()


        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Config section (Section 1)
        config_widget = QWidget()
        config_widget.setFixedWidth(300)
        config_layout = QVBoxLayout()
        config_widget.setLayout(config_layout)

        self.logpath = QLineEdit()
        self.logpath.setText("pm_data/running_example.xes")
        config_layout.addWidget(QLabel("Log Path:"))
        config_layout.addWidget(self.logpath)
        
        self.serialize_pop = QCheckBox("Serialize Population")
        config_layout.addWidget(self.serialize_pop)
        
        self.time_execution = QCheckBox("Time Execution")
        config_layout.addWidget(self.time_execution)
        
        self.stop_after = QSpinBox()
        self.stop_after.setRange(1, 10000)
        self.stop_after.setValue(500)
        self.stop_after.valueChanged.connect(self.update_info_box)
        config_layout.addWidget(QLabel("Number of Generations:"))
        config_layout.addWidget(self.stop_after)
        
        self.setup_map = {}
        self.setup_runs = QSpinBox()
        self.setup_runs.setRange(1, 1000)
        self.setup_runs.setValue(4)
        self.setup_runs.valueChanged.connect(self.update_info_box)
        config_layout.addWidget(QLabel("Number of Setup Runs:"))
        config_layout.addWidget(self.setup_runs)
        
        self.config_description = QTextEdit()
        config_layout.addWidget(QLabel("Config Description:"))
        config_layout.addWidget(self.config_description)
        
        self.info_box = QLabel()
        config_layout.addWidget(self.info_box)
        
        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.clicked.connect(self.save_config)
        config_layout.addWidget(self.save_config_button)
        
        # Middle section (Section 2)
        middle_widget = QWidget()
        middle_layout = QVBoxLayout()
        middle_widget.setLayout(middle_layout)
        self.tree_widget = QSvgWidget()
        self.tree_widget.setFixedSize(1420,700)
        middle_layout.addWidget(self.tree_widget)
        
        # Parameters section
        params_widget = QWidget()
        params_layout = QVBoxLayout()
        params_widget.setLayout(params_layout)
        
        self.load_params_button = QPushButton("Load Params")
        self.load_params_button.clicked.connect(self.load_params)
        params_layout.addWidget(self.load_params_button)
        
        self.base_params_info = QLabel()
        params_layout.addWidget(self.base_params_info)
        
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_content = QWidget()
        self.params_content_layout = QVBoxLayout()
        self.params_content.setLayout(self.params_content_layout)
        self.params_scroll.setWidget(self.params_content)
        params_layout.addWidget(self.params_scroll)
        
        param_buttons_layout = QHBoxLayout()
        self.add_param_button = QPushButton("Add New Parameter")
        self.add_param_button.clicked.connect(self.add_parameter)
        param_buttons_layout.addWidget(self.add_param_button)
        
        self.remove_param_button = QPushButton("Remove Parameter")
        self.remove_param_button.clicked.connect(self.remove_parameter)
        param_buttons_layout.addWidget(self.remove_param_button)
        
        params_layout.addLayout(param_buttons_layout)
        
        middle_layout.addWidget(params_widget)
        
        # Setup list section (Section 3)
        setup_list_widget = QWidget()
        setup_list_widget.setFixedWidth(90)
        setup_list_layout = QVBoxLayout()
        setup_list_widget.setLayout(setup_list_layout)
        
        self.setup_list = QListWidget()
        self.setup_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        setup_list_layout.addWidget(self.setup_list)
        
        # Set up splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(config_widget)
        splitter.addWidget(middle_widget)
        splitter.addWidget(setup_list_widget)
        
        # Set size policies
        config_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        middle_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        setup_list_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        self.base_params = {}
        
        # Load default params
        self.load_params("params/default_params.json")
        self.add_parameter()
        self.update_info_box()


        
    def load_params(self, file_path=None):
        if not file_path:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Params File", "", "JSON Files (*.json)")
        else:
            file_name = file_path
        
        if file_name:
            with open(file_name, 'r') as f:
                self.base_params = json.load(f)
            self.update_info_box()
            self.base_params_info.setText(f"Currently loaded: {file_name}")
    

    def add_parameter(self):
        param_widget = QWidget()
        param_layout = QHBoxLayout()
        
        param_key = QComboBox()
        param_key.addItems(self.get_nested_keys(self.base_params))
        param_key.activated.connect(self.update_tree)
        param_layout.addWidget(param_key)
        
        param_values = QLineEdit()
        param_values.editingFinished.connect(self.update_tree)
        param_values.editingFinished.connect(self.update_info_box)
        param_layout.addWidget(param_values)
        
        param_widget.setLayout(param_layout)
        self.params_content_layout.addWidget(param_widget)
        self.update_tree()


    def remove_parameter(self):
        if self.params_content_layout.count() > 0:
            item = self.params_content_layout.takeAt(self.params_content_layout.count() - 1)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            self.update_tree()


    def update_tree(self):
        param_values = self.get_param_values()
        if not param_values:
            return

        dot = graphviz.Digraph(comment='Parameter Tree')
        # dot.attr(rankdir='TB')  # Top to Bottom layout
        dot.attr(rankdir='LR')  # Top to Bottom layout

        setup_counter = 1
        self.setup_map = {}

        def create_node_label(params, is_leaf=False):
            if is_leaf:
                label = f"<B>Setup {setup_counter}</B><BR/>"
            else:
                label = ""
            label += "<BR/>".join([f"{k.lstrip("metric_dict.")}: {v}" for k, v in params.items()])
            return f"<{label}>"

        def add_nodes(current_params, remaining_params, parent_id=None):
            nonlocal setup_counter

            if not remaining_params:
                # This is a leaf node (complete setup)
                node_id = f"setup_{setup_counter}"
                dot.node(node_id, create_node_label(current_params, is_leaf=True), shape="box", width="0", height="0", fixedsize="false")
                if parent_id:
                    dot.edge(parent_id, node_id)
                self.setup_map[setup_counter] = current_params
                setup_counter += 1
                return

            current_key, current_values = next(iter(remaining_params.items()))
            new_remaining = dict(list(remaining_params.items())[1:])

            for value in current_values:
                new_params = {**current_params, current_key: value}
                node_id = f"node_{'_'.join(map(str, new_params.values()))}"
                
                dot.node(node_id, create_node_label({current_key: value}), shape="box", width="0", height="0", fixedsize="false")

                if parent_id:
                    dot.edge(parent_id, node_id)

                add_nodes(new_params, new_remaining, node_id)

        # Add root node
        root_id = "root"
        dot.node(root_id, "")

        # Start the recursion with the root node as parent
        add_nodes({}, param_values, root_id)

        # Render the tree
        dot.attr(size='1420,700')
        dot.attr('node', fontsize='10')
        dot.render('tree', format='svg', cleanup=True)

        # Display the tree
        self.tree_widget.load('tree.svg')

        # Update checkboxes
        self.update_setup_list()


    def get_param_values(self):
        param_values = {}
        for i in range(self.params_content_layout.count()):
            widget = self.params_content_layout.itemAt(i).widget()
            key = widget.layout().itemAt(0).widget().currentText()
            values = widget.layout().itemAt(1).widget().text().split(';')
            param_values[key] = values
        return param_values


    def update_setup_list(self):
        self.setup_list.clear()
        for setup_num in self.setup_map.keys():
            item = QListWidgetItem(f"Setup {setup_num}")
            self.setup_list.addItem(item)

        for index in range(self.setup_list.count()):
            self.setup_list.item(index).setSelected(True)



    def get_nested_keys(self, d, prefix=''):
        keys = []
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                keys.extend(self.get_nested_keys(v, new_key))
            else:
                keys.append(new_key)
        return keys
    

    def save_config(self):
        name, ok = QInputDialog.getText(self, "Save Config", "Enter config name:")
        if ok and name:
            config = {
                "name": name,
                "setups": []
            }
            
            config_dir = f"configs/{name}"
            os.makedirs(config_dir, exist_ok=True)
            
            selected_setups = [int(item.text().split()[1]) for item in self.setup_list.selectedItems()]
            
            for setup_num, params in self.setup_map.items():
                if setup_num in selected_setups:
                    setup = {
                        "setupname": f"setup_{setup_num}",
                        "parampath": f"{config_dir}/setup_{setup_num}.json",
                        "logpath": self.logpath.text(),
                        "ga_kwargs": {
                            "is_pop_serialized": self.serialize_pop.isChecked(),
                            "is_timed": self.time_execution.isChecked()
                        },
                        "stop_cond": {
                            "var": "gen",
                            "val": self.stop_after.value()
                        },
                        "n_runs": self.setup_runs.value(),
                        "send_gen_info_to_console": False,
                        "is_profiled": False
                    }
                    config["setups"].append(setup)
            
            # write the config file
            with open(f"{config_dir}/{name}.json", 'w') as f:
                json.dump(config, f, indent=4)

            # write the param files
            for i in selected_setups:
                with open(f"{config_dir}/setup_{i}.json", 'w') as f:
                    json.dump(self.generate_param_dict(self.setup_map[i]), f, indent=4)

            # write the change file
            with open(f"{config_dir}/param_changes.txt", 'w') as f:
                f.write(f"Config Description:\n{self.config_description.toPlainText()}\n\n")
                for setup_num in selected_setups:
                    f.write(f"Setup {setup_num}:\n")
                    for key, value in self.setup_map[setup_num].items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
            
            self.show_notification("Config saved successfully!")
    
    def generate_param_dict(self, param_combo: dict):
        new_params = self.base_params.copy()
        for key, value in param_combo.items():
            parsed_value = json.loads(value)
            # traverse key path
            if "." in key:
                key_path = key.split(".")
                inner_dict = new_params
                for key in key_path[:-1]:
                    inner_dict = inner_dict[key]
                inner_dict[key_path[-1]] = parsed_value
            # can assign directly
            else:
                new_params[key] = parsed_value
        
        return new_params
    

    def update_info_box(self):
        num_setups = max(1, len(self.setup_map))

        num_runs = num_setups * self.setup_runs.value()
        total_generations = num_runs * self.stop_after.value()
        
        info_text = f"Number of setups: {num_setups}\n"
        info_text += f"Total runs: {num_runs}\n"
        info_text += f"Total generations: {total_generations}"
        
        self.info_box.setText(info_text)
    

    def show_notification(self, message):
        notification = QMessageBox(self)
        notification.setText(message)
        notification.setWindowTitle("Notification")
        notification.setStandardButtons(QMessageBox.StandardButton.NoButton)
        notification.show()
        QTimer.singleShot(500, notification.accept)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())