#include "settings_page_registry.h"

#include <QCoreApplication>
#include <QMetaObject>
#include <QThread>

namespace pc::ui {

SettingsPageRegistry* SettingsPageRegistry::instance()
{
    // One instance for the process; owned by Qt (no delete)
    static SettingsPageRegistry* s_instance = new SettingsPageRegistry(qApp);
    return s_instance;
}

SettingsPageRegistry::SettingsPageRegistry(QObject* parent)
    : QAbstractListModel(parent)
{
}

int SettingsPageRegistry::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid())
        return 0;
    return static_cast<int>(m_pages.size());
}

QVariant SettingsPageRegistry::data(const QModelIndex& index, int role) const
{
    if (!index.isValid())
        return {};
    const int row = index.row();
    if (row < 0 || row >= rowCount())
        return {};

    const PageEntry& e = m_pages[static_cast<size_t>(row)];
    switch (static_cast<Role>(role)) {
    case Role::Key:     return e.key;
    case Role::Title:   return e.title;
    case Role::PageUrl: return e.pageUrl;
    }
    return {};
}

QHash<int, QByteArray> SettingsPageRegistry::roleNames() const
{
    QHash<int, QByteArray> roles;
    roles[static_cast<int>(Role::Key)] = "key";
    roles[static_cast<int>(Role::Title)] = "title";
    roles[static_cast<int>(Role::PageUrl)] = "pageUrl";
    return roles;
}

int SettingsPageRegistry::findIndexByKey(const QString& key) const
{
    for (int i = 0; i < rowCount(); ++i) {
        if (m_pages[static_cast<size_t>(i)].key == key)
            return i;
    }
    return -1;
}

int SettingsPageRegistry::indexOfKey(const QString& key) const
{
    return findIndexByKey(key);
}

QUrl SettingsPageRegistry::pageUrlAt(int index) const
{
    if (index < 0 || index >= rowCount())
        return {};
    return m_pages[static_cast<size_t>(index)].pageUrl;
}

static bool isOnObjectThread(QObject* obj)
{
    return QThread::currentThread() == obj->thread();
}

bool SettingsPageRegistry::addPage(const QString& key, const QString& title, const QUrl& pageUrl)
{
    if (!isOnObjectThread(this)) {
        bool result = false;
        QMetaObject::invokeMethod(
            this,
            [this, &result, key, title, pageUrl]() {
                result = this->addPage(key, title, pageUrl);
            },
            Qt::BlockingQueuedConnection);
        return result;
    }

    if (key.isEmpty())
        return false;

    const int idx = findIndexByKey(key);
    if (idx < 0) {
        const int insertAt = rowCount();
        beginInsertRows(QModelIndex{}, insertAt, insertAt);
        m_pages.push_back(PageEntry{key, title, pageUrl});
        endInsertRows();
        return true;
    }

    PageEntry& existing = m_pages[static_cast<size_t>(idx)];
    existing.title = title;
    existing.pageUrl = pageUrl;

    const QModelIndex modelIndex = index(idx);
    emit dataChanged(modelIndex, modelIndex,
                     { static_cast<int>(Role::Title), static_cast<int>(Role::PageUrl) });
    return true;
}

bool SettingsPageRegistry::removePage(const QString& key)
{
    if (!isOnObjectThread(this)) {
        bool result = false;
        QMetaObject::invokeMethod(
            this,
            [this, &result, key]() {
                result = this->removePage(key);
            },
            Qt::BlockingQueuedConnection);
        return result;
    }

    const int idx = findIndexByKey(key);
    if (idx < 0)
        return false;

    beginRemoveRows(QModelIndex{}, idx, idx);
    m_pages.removeAt(idx);
    endRemoveRows();
    return true;
}

void SettingsPageRegistry::clear()
{
    if (!isOnObjectThread(this)) {
        QMetaObject::invokeMethod(this, [this]() { this->clear(); }, Qt::QueuedConnection);
        return;
    }

    beginResetModel();
    m_pages.clear();
    endResetModel();
}

}